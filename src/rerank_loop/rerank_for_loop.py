import random
import numpy
import json
import time
import argparse
import os
import shutil
import uuid
import torch
import torch.distributed as dist
from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoModelForSequenceClassification, AutoTokenizer
# import from ../../externals/UPR/utils
import sys
sys.path.append('../../externals/UPR')
#import ElasticSearchBM25Retriever from ../retrieval_loop/elastic_bm25_search_with_metadata.py
sys.path.append('../retrieval_loop')
sys.path.append('../../externals/MonoT5')

from utils import print_rank_0
from utils.openqa_dataset import get_openqa_dataset, get_one_epoch_dataloader
from utils.initialize import initialize_distributed
from elastic_bm25_search_with_metadata import ElasticSearchBM25Retriever
from evaluate_dpr_retrieval import evaluate_retrieval
from monot5_support import MonoT5, Reranker, Query, Text
from rankgpt_support import get_openai_api, sliding_windows
import logging

# 禁用日志记录
logging.disable(logging.INFO)

def set_random_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)


class UnsupervisedPassageReranker():
    def __init__(self, args):
        self.model = None
        self.dataloader = None
        self.dataset = None
        self.evidence_dataset = None

        self.args = args
        self.log_interval = args.log_interval
        # Hard coding the per gpu batch size to 1
        self.batch_size = 1

        self.load_attributes()
        self.is_main_builder = dist.get_rank() == 0
        self.num_total_builders = dist.get_world_size()
        # Create a temporary directory using uuid for saving the shards
        # uuid is used to avoid conflicts when multiple processes are running
        uid = str(uuid.uuid4())
        self.temp_dir_name = os.path.join(args.reranker_output_dir, '_tmp_reranker_{}'.format(uid))

    def load_attributes(self):
        print_rank_0("Loading {} weights".format(self.args.hf_model_name))
        self.tokenizer = T5Tokenizer.from_pretrained(self.args.hf_model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(self.args.hf_model_name,
                                                                torch_dtype=torch.bfloat16 if self.args.use_bf16 else torch.float32)

        for param in self.model.parameters():
            param.requires_grad = False

        if self.args.use_gpu:
            self.model = self.model.cuda()

        print_rank_0("Loaded {} weights".format(self.args.hf_model_name))

        # disable dropout
        self.model.eval()
        print_rank_0('Please make sure you have started elastic search server')

        self.index = ElasticSearchBM25Retriever.create(self.args.elasticsearch_url, self.args.index_name)

        self.dataset = get_openqa_dataset(self.args.task_name,
                                          self.args.retriever_topk_passages_path,
                                          sample_rate=self.args.sample_rate)

        self.dataloader = iter(get_one_epoch_dataloader(self.dataset,
                                                        self.args,
                                                        self.batch_size))
        self.iteration = self.total_processed = 0

    def track_and_report_progress(self, batch_size):
        """Utility function for tracking progress"""
        self.iteration += 1
        self.total_processed += batch_size * self.num_total_builders
        if self.is_main_builder and self.iteration % self.log_interval == 0:
            print('Batch {:10d} | Total {:10d}'.format(self.iteration, self.total_processed), flush=True)

    def do_inference(self):
        reranked_answers_list = []
        original_answers_list = []
        reranked_data = {}

        start_time = time.time()

        while True:
            try:
                # batch also has query_tokens and query_pad_data
                batch = next(self.dataloader)
            except (StopIteration, IndexError):
                break

            assert len(batch['id']) == 1, "Currently, we are doing inference with batch size 1"

            all_contexts = batch['encoder_ids'][0][:self.args.topk_passages]

            all_ids = []
            has_answer_list = []

            for i, context in enumerate(all_contexts):
                doc_id = context.get("docid")
                doc = self.index.get_document_by_id(doc_id).page_content
                if len(doc.split('\n')) < 2:
                    title = ''
                    text = doc
                else:
                    title, text = doc.split('\n', 1)
                ids = "{} {} {}. {}".format(self.args.verbalizer_head, title, text, self.args.verbalizer)
                all_ids.append(ids)
                has_answer_list.append(context.get('has_answer'))

            input_encoding = self.tokenizer(all_ids,
                                            padding='longest',
                                            max_length=512,
                                            pad_to_multiple_of=8,
                                            truncation=True,
                                            return_tensors='pt')

            context_tensor, attention_mask = input_encoding.input_ids, input_encoding.attention_mask
            if self.args.use_gpu:
                context_tensor = context_tensor.cuda()
                attention_mask = attention_mask.cuda()

            decoder_prefix = batch['decoder_ids']
            target_encoding = self.tokenizer(decoder_prefix,
                                             max_length=128,
                                             truncation=True,
                                             return_tensors='pt')

            decoder_prefix_tensor = target_encoding.input_ids
            if self.args.use_gpu:
                decoder_prefix_tensor = decoder_prefix_tensor.cuda()

            decoder_prefix_tensor = torch.repeat_interleave(decoder_prefix_tensor,
                                                            len(context_tensor),
                                                            dim=0)
            sharded_nll_list = []

            for i in range(0, len(context_tensor), self.args.shard_size):
                encoder_tensor_view = context_tensor[i: i + self.args.shard_size]
                attention_mask_view = attention_mask[i: i + self.args.shard_size]
                decoder_tensor_view = decoder_prefix_tensor[i: i + self.args.shard_size]
                with torch.no_grad():
                    logits = self.model(input_ids=encoder_tensor_view,
                                        attention_mask=attention_mask_view,
                                        labels=decoder_tensor_view).logits

                log_softmax = torch.nn.functional.log_softmax(logits, dim=-1)
                nll = -log_softmax.gather(2, decoder_tensor_view.unsqueeze(2)).squeeze(2)

                avg_nll = torch.sum(nll, dim=1)
                sharded_nll_list.append(avg_nll)

            topk_scores, indexes = torch.topk(-torch.cat(sharded_nll_list), k=len(context_tensor))
            indexes = indexes.cpu()
            ranked_answers = torch.BoolTensor(has_answer_list)[indexes]

            # Save the essential information to be used for saving the re-ranked information component.
            original_answers_list.append(has_answer_list)
            reranked_answers_list.append(ranked_answers.tolist())

            reordered_context = [all_contexts[i] for i in indexes]

            for i, ctx in enumerate(reordered_context):
                ctx['score'] = topk_scores[i].item()

            item = {
                    "question": batch['question'][0],
                    "answers": batch['answers'][0],
                    "contexts": reordered_context[:self.args.report_topk_accuracies[-1]]}
            reranked_data[batch['query_id'][0]] = item

            self.track_and_report_progress(batch_size=len(batch['id']))

        end_time = time.time()
        time_taken = (end_time - start_time) / len(reranked_data)
        torch.distributed.barrier()

        print_rank_0("Time taken: {} seconds".format(time_taken))

        self.compute_topk_recall(original_answers_list, string_prefix="Original Ranking")
        self.compute_topk_recall(reranked_answers_list, string_prefix="Re-Ranking")

        if self.args.merge_shards_and_save:
            self.save_shard(reranked_data)

        del self.model
        # This process signals to finalize its shard and then synchronize with the other processes
        torch.distributed.barrier()

        if self.args.merge_shards_and_save:
            # rank 0 process builds the final copy
            if self.is_main_builder:
                self.merge_shards_and_save()
            # complete building the final copy
            torch.distributed.barrier()

    @staticmethod
    def calculate_topk_hits(scores, max_k):
        top_k_hits = [0] * max_k
        for question_hits in scores:
            best_hit = next((i for i, x in enumerate(question_hits[:max_k]) if x), None)
            if best_hit is not None:
                top_k_hits[best_hit:] = [v + 1 for v in top_k_hits[best_hit:]]
        return top_k_hits

    def compute_topk_recall(self, answers_list, string_prefix):
        topk_hits = self.calculate_topk_hits(answers_list, max_k=self.args.report_topk_accuracies[-1])

        topk_hits = torch.FloatTensor(topk_hits).cuda()
        n_docs = torch.FloatTensor([len(answers_list)]).cuda()
        torch.distributed.all_reduce(topk_hits, torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(n_docs, torch.distributed.ReduceOp.SUM)

        if torch.distributed.get_rank() == 0:
            topk_hits = topk_hits / n_docs
            print(string_prefix)
            for i in self.args.report_topk_accuracies:
                print_rank_0("top-{}: {:.2f}".format(i, topk_hits[i - 1] * 100))
            print("\n")

    def save_shard(self, data):
        """
        Save the block data that was created this in this process
        """
        if not os.path.isdir(self.temp_dir_name):
            os.makedirs(self.temp_dir_name, exist_ok=True)

        outpath = os.path.join(self.temp_dir_name, "rank{}.json".format(dist.get_rank()))
        with open(outpath, "w", encoding='utf-8') as writer:
            writer.write(json.dumps(data, indent=4, ensure_ascii=False) + "\n")

    def merge_shards_and_save(self):
        """Combine all the shards made using self.save_shard()"""
        shard_names = os.listdir(self.temp_dir_name)
        all_data = {}

        for fname in os.listdir(self.temp_dir_name):
            shard_size = 0
            old_size = len(all_data)
            fpath = '{}/{}'.format(self.temp_dir_name, fname)
            with open(fpath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                shard_size = len(data)
                all_data.update(data)

            assert len(all_data) == old_size + shard_size
            os.remove(fpath)

        # save the consolidated shards
        outpath = os.path.join(self.args.reranker_output_dir, "{}.json".format(self.args.special_suffix))
        outpath_trec = os.path.join(self.args.reranker_output_dir, "{}.trec".format(self.args.special_suffix))

        with open(outpath, 'w', encoding='utf-8') as writer:
            writer.write(json.dumps(all_data, indent=4, ensure_ascii=False) + "\n")

        print("Computing top-k accuracies for the merged data")
        evaluate_retrieval(outpath, [20, 100], False)

        with open(outpath_trec, 'w', encoding='utf-8') as writer:
            for qid, item in all_data.items():
                for i, ctx in enumerate(item['contexts']):
                    writer.write("{} Q0 {} {} {} bm25_{}\n".format(
                        qid, ctx['docid'], i+1, ctx['score'], self.args.method))

        print("Finished merging {} shards for a total of {} embeds".format(
            len(shard_names), len(all_data)), flush=True)

        # make sure that every single piece of data was embedded
        assert len(all_data) == len(self.dataset)

        shutil.rmtree(self.temp_dir_name, ignore_errors=True)


class MonoT5Reranker(UnsupervisedPassageReranker):
    def __init__(self, args):
        super().__init__(args)

    def load_attributes(self):
        print_rank_0("Loading {} weights".format(self.args.hf_model_name))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = T5ForConditionalGeneration.from_pretrained(self.args.hf_model_name,
                                                            torch_dtype=torch.bfloat16 if self.args.use_bf16 else torch.float32)
        model = model.to(device)
        self.model = MonoT5(model=model)

        print_rank_0("Loaded {} weights".format(self.args.hf_model_name))

        print_rank_0('Please make sure you have started elastic search server')

        self.index = ElasticSearchBM25Retriever.create(self.args.elasticsearch_url, self.args.index_name)

        self.dataset = get_openqa_dataset(self.args.task_name,
                                          self.args.retriever_topk_passages_path,
                                          sample_rate=self.args.sample_rate)

        self.dataloader = iter(get_one_epoch_dataloader(self.dataset,
                                                        self.args,
                                                        self.batch_size))
        self.iteration = self.total_processed = 0

    def do_inference(self):
        reranked_answers_list = []
        original_answers_list = []
        reranked_data = {}

        start_time = time.time()

        while True:
            try:
                # batch also has query_tokens and query_pad_data
                batch = next(self.dataloader)
            except (StopIteration, IndexError):
                break

            assert len(batch['id']) == 1, "Currently, we are doing inference with batch size 1"

            all_contexts = batch['encoder_ids'][0][:self.args.topk_passages]

            passages = []
            has_answer_list = []

            for i, context in enumerate(all_contexts):
                pid = context.get("docid")
                passage = self.index.get_document_by_id(pid).page_content
                passages.append([pid, passage])
                has_answer_list.append(context.get('has_answer'))
            old_docids = [p[0] for p in passages]
            texts = [Text(p[1], {'docid': p[0]}, 0) for p in passages]
            query = Query(batch['question'][0])

            reranked = self.model.rerank(query, texts)
            ranked_docids = [r.metadata['docid'] for r in reranked]
            ranked_scores = [r.score for r in reranked]

            indexes = [old_docids.index(did) for did in ranked_docids]
            topk_scores = torch.FloatTensor(ranked_scores)
            ranked_answers = torch.BoolTensor(has_answer_list)[indexes]

            # Save the essential information to be used for saving the re-ranked information component.
            original_answers_list.append(has_answer_list)
            reranked_answers_list.append(ranked_answers.tolist())

            reordered_context = [all_contexts[i] for i in indexes]

            for i, ctx in enumerate(reordered_context):
                ctx['score'] = topk_scores[i].item()

            item = {
                "question": batch['question'][0],
                "answers": batch['answers'][0],
                "contexts": reordered_context[:self.args.report_topk_accuracies[-1]]}
            reranked_data[batch['query_id'][0]] = item

            self.track_and_report_progress(batch_size=len(batch['id']))

        end_time = time.time()
        time_taken = (end_time - start_time) / len(reranked_data)
        torch.distributed.barrier()

        print_rank_0("Time taken: {} seconds".format(time_taken))

        self.compute_topk_recall(original_answers_list, string_prefix="Original Ranking")
        self.compute_topk_recall(reranked_answers_list, string_prefix="Re-Ranking")

        if self.args.merge_shards_and_save:
            self.save_shard(reranked_data)

        del self.model
        # This process signals to finalize its shard and then synchronize with the other processes
        torch.distributed.barrier()

        if self.args.merge_shards_and_save:
            # rank 0 process builds the final copy
            if self.is_main_builder:
                self.merge_shards_and_save()
            # complete building the final copy
            torch.distributed.barrier()


class BGEReranker(UnsupervisedPassageReranker):
    def __init__(self, args):
        super().__init__(args)

    def load_attributes(self):
        print_rank_0("Loading {} weights".format(self.args.hf_model_name))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(self.args.hf_model_name)
        model = AutoModelForSequenceClassification.from_pretrained(self.args.hf_model_name, torch_dtype='auto')
        model.eval()
        self.model = model.to(device)
        for param in self.model.parameters():
            param.requires_grad = False

        print_rank_0("Loaded {} weights".format(self.args.hf_model_name))

        print_rank_0('Please make sure you have started elastic search server')

        self.index = ElasticSearchBM25Retriever.create(self.args.elasticsearch_url, self.args.index_name)

        self.dataset = get_openqa_dataset(self.args.task_name,
                                          self.args.retriever_topk_passages_path,
                                          sample_rate=self.args.sample_rate)

        self.dataloader = iter(get_one_epoch_dataloader(self.dataset,
                                                        self.args,
                                                        self.batch_size))
        self.iteration = self.total_processed = 0

    def do_inference(self):
        """
        model usage:
        pairs = [['what is panda?', 'hi'], ['what is panda?', 'The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.']]
        with torch.no_grad():
            inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
            scores = model(**inputs, return_dict=True).logits.view(-1, ).float()
        print(scores)
        :return:
        """
        reranked_answers_list = []
        original_answers_list = []
        reranked_data = {}

        start_time = time.time()

        while True:
            try:
                # batch also has query_tokens and query_pad_data
                batch = next(self.dataloader)
            except (StopIteration, IndexError):
                break

            assert len(batch['id']) == 1, "Currently, we are doing inference with batch size 1"

            all_contexts = batch['encoder_ids'][0][:self.args.topk_passages]

            passages = []
            has_answer_list = []

            for i, context in enumerate(all_contexts):
                pid = context.get("docid")
                passage = self.index.get_document_by_id(pid).page_content
                passages.append([pid, passage])
                has_answer_list.append(context.get('has_answer'))

            pairs = [[batch['question'][0], p[1]] for p in passages]
            with torch.no_grad():
                inputs = self.tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
                inputs = inputs.to(self.model.device)
                scores = self.model(**inputs, return_dict=True).logits.view(-1, ).float()
            ranked_scores = scores.tolist()

            indexes = numpy.argsort(ranked_scores)
            indexes = indexes[::-1].copy()
            topk_scores = torch.FloatTensor(ranked_scores)

            ranked_answers = torch.BoolTensor(has_answer_list)[indexes]

            # Save the essential information to be used for saving the re-ranked information component.
            original_answers_list.append(has_answer_list)
            reranked_answers_list.append(ranked_answers.tolist())

            reordered_context = [all_contexts[i] for i in indexes]

            for i, ctx in enumerate(reordered_context):
                ctx['score'] = topk_scores[i].item()

            item = {
                "question": batch['question'][0],
                "answers": batch['answers'][0],
                "contexts": reordered_context[:self.args.report_topk_accuracies[-1]]}
            reranked_data[batch['query_id'][0]] = item

            self.track_and_report_progress(batch_size=len(batch['id']))

        end_time = time.time()
        time_taken = (end_time - start_time) / len(reranked_data)
        torch.distributed.barrier()

        print_rank_0("Time taken: {} seconds".format(time_taken))

        self.compute_topk_recall(original_answers_list, string_prefix="Original Ranking")
        self.compute_topk_recall(reranked_answers_list, string_prefix="Re-Ranking")

        if self.args.merge_shards_and_save:
            self.save_shard(reranked_data)

        del self.model
        # This process signals to finalize its shard and then synchronize with the other processes
        torch.distributed.barrier()

        if self.args.merge_shards_and_save:
            # rank 0 process builds the final copy
            if self.is_main_builder:
                self.merge_shards_and_save()
            # complete building the final copy
            torch.distributed.barrier()



class RankGPTReranker(UnsupervisedPassageReranker):
    def __init__(self, args):
        super().__init__(args)

    def load_attributes(self):
        print_rank_0("Loading {} weights".format(self.args.hf_model_name))

        print_rank_0('Please make sure you have started elastic search server')

        self.model = self.args.rankgpt_llm_model_name

        get_openai_api(self.model)

        self.prompter_name = self.args.rankgpt_prompter_name

        self.rank_end = self.args.rankgpt_rank_end

        self.window_size = self.args.rankgpt_window_size

        self.step_size = self.args.rankgpt_step_size

        self.index = ElasticSearchBM25Retriever.create(self.args.elasticsearch_url, self.args.index_name)

        self.dataset = get_openqa_dataset(self.args.task_name,
                                          self.args.retriever_topk_passages_path,
                                          sample_rate=self.args.sample_rate)

        self.dataloader = iter(get_one_epoch_dataloader(self.dataset,
                                                        self.args,
                                                        self.batch_size))
        self.iteration = self.total_processed = 0

    def do_inference(self):
        """
        model usage:
        pairs = [['what is panda?', 'hi'], ['what is panda?', 'The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.']]
        with torch.no_grad():
            inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
            scores = model(**inputs, return_dict=True).logits.view(-1, ).float()
        print(scores)
        :return:
        """
        reranked_answers_list = []
        original_answers_list = []
        reranked_data = {}

        start_time = time.time()

        while True:
            try:
                # batch also has query_tokens and query_pad_data
                batch = next(self.dataloader)
            except (StopIteration, IndexError):
                break

            assert len(batch['id']) == 1, "Currently, we are doing inference with batch size 1"

            all_contexts = batch['encoder_ids'][0][:self.args.topk_passages]

            passages = []
            has_answer_list = []

            for i, context in enumerate(all_contexts):
                pid = context.get("docid")
                passage = self.index.get_document_by_id(pid).page_content
                passages.append([pid, passage])
                has_answer_list.append(context.get('has_answer'))

            # ranking:[{'query': query,
            #         'hits': [{
            #                   'content': content,
            #                   'qid': qid,
            #                   'docid': docid,
            #                   'rank': rank,
            #                   'score': score
            #                   }, ...]
            #                   }, ...]
            ranking = {'query': batch['question'][0],
                        'hits': [{'content': p[1],
                                  'qid': batch['query_id'][0],
                                  'docid': p[0],
                                  'rank': i+1,
                                  'score': float(len(all_contexts)-i)} for i, p in enumerate(passages)]
                        }
            new_ranking = sliding_windows(ranking, rank_start=0, rank_end=self.rank_end, window_size=self.window_size,
                                          step=self.step_size, model=self.model, prompter_name=self.prompter_name)
            new_ranking = new_ranking['hits']
            ranked_docids = [r['docid'] for r in new_ranking]
            ranked_scores = [r['score'] for r in new_ranking]
            old_docids = [p[0] for p in passages]

            indexes = [old_docids.index(did) for did in ranked_docids]
            topk_scores = torch.FloatTensor(ranked_scores)
            ranked_answers = torch.BoolTensor(has_answer_list)[indexes]

            # Save the essential information to be used for saving the re-ranked information component.
            original_answers_list.append(has_answer_list)
            reranked_answers_list.append(ranked_answers.tolist())

            reordered_context = [all_contexts[i] for i in indexes]

            for i, ctx in enumerate(reordered_context):
                ctx['score'] = topk_scores[i].item()

            item = {
                "question": batch['question'][0],
                "answers": batch['answers'][0],
                "contexts": reordered_context[:self.args.report_topk_accuracies[-1]]}
            reranked_data[batch['query_id'][0]] = item

            self.track_and_report_progress(batch_size=len(batch['id']))

        end_time = time.time()
        time_taken = (end_time - start_time) / len(reranked_data)
        torch.distributed.barrier()

        print_rank_0("Time taken: {} seconds".format(time_taken))

        self.compute_topk_recall(original_answers_list, string_prefix="Original Ranking")
        self.compute_topk_recall(reranked_answers_list, string_prefix="Re-Ranking")

        if self.args.merge_shards_and_save:
            self.save_shard(reranked_data)

        del self.model
        # This process signals to finalize its shard and then synchronize with the other processes
        torch.distributed.barrier()

        if self.args.merge_shards_and_save:
            # rank 0 process builds the final copy
            if self.is_main_builder:
                self.merge_shards_and_save()
            # complete building the final copy
            torch.distributed.barrier()




def get_args():

    parser = argparse.ArgumentParser()

    group = parser.add_argument_group(title='argument-parser')

    group.add_argument('--method', type=str, default='UPR',)

    group.add_argument('--config', type=str, default='config.json',
                       help='Path to the configuration file, which can overlap the cmd and default value.')

    group.add_argument('--local_rank', type=int, default=-1,
                       help='local rank passed from distributed launcher.')

    group.add_argument('--main-port', type=int, default=29500,
                       help='Main port number.')

    group.add_argument('--special-suffix', type=str, default="",
                       help='special suffix extension for saving merged file')

    group.add_argument('--retriever-topk-passages-path', type=str, default="downloads/data/retriever-outputs/nq-dev.json",
                       help='Path of the Top-K passage output file from retriever (.json file)')

    group.add_argument('--topk-passages', type=int, default=1000,
                       help='number of topk passages to select')

    group.add_argument('--log-interval', type=int, default=100,
                       help='Interval between progress updates')

    group.add_argument('--shard-size', type=int, default=16)

    group.add_argument('--num-workers', type=int, default=2,
                       help="Dataloader number of workers.")

    group.add_argument('--reranker-output-dir', type=str, default="downloads/data/retriever-outputs/",
                       help='Path to save UPR results')

    group.add_argument('--task-name', type=str, default="reranking",
                       help='Name of the task.')

    group.add_argument('--hf-model-name', type=str, default="t5-large",
                       help='Name of the HF model.')

    group.add_argument('--interactive-node', action='store_true',
                       help='If the node is interactive or not')

    group.add_argument('--use-gpu', action='store_true',
                       help='Use GPU or not')

    group.add_argument('--use-bf16', action='store_true',
                       help='Whether to use BF16 data format for the T0/T5 models')

    group.add_argument('--merge-shards-and-save', action='store_true',
                       help='whether to merge individual data shards or not for reranking')

    group.add_argument('--sample-rate', type=float, default=1.,
                       help="Sample rate for the number of examples.")

    group.add_argument('--random-seed', type=int, default=1234,
                       help="Random seed.")

    group.add_argument('--index-name', type=str, default="nq",
                       help="Name of the index in elastic search")

    group.add_argument('--elasticsearch-url', type=str, default="http://0.0.0.0:9978",
                          help="URL of the elastic search server")

    group.add_argument('--verbalizer', type=str, default="Please write a question based on this passage.",
                       help='Prompt string for generating the target tokens')

    group.add_argument('--verbalizer-head', type=str, default="Passage: ",
                       help='The string token used to represent encoder input')

    group.add_argument('--report-topk-accuracies', nargs='+', type=int, default=[1, 5, 10, 20, 50, 100],
                       help="Which top-k accuracies to report (e.g. '1 5 20')")

    group.add_argument('--rankgpt-llm-model-name', type=str, default="gpt-3.5-turbo",
                       help='Name of the LLM ranking model, options: Qwen, Llama, chatglm3, baichuan2-13b-chat, gpt-3.5-turbo.')

    group.add_argument('--rankgpt-prompter-name', type=str, default="LLMreranker",
                       help='Name of the prompt template.')

    group.add_argument('--rankgpt-rank-end', type=int, default=100,
                       help='Ranking end.')

    group.add_argument('--rankgpt-window-size', type=int, default=20,
                          help='Window size.')

    group.add_argument('--rankgpt-step-size', type=int, default=10,
                            help='Step size.')


    args = parser.parse_args()
    args.keep_empty = False

    # some default/dummy values for the tokenizer
    # Distributed args.
    args.rank = int(os.getenv('RANK', '0'))
    args.world_size = int(os.getenv("WORLD_SIZE", '1'))

    # 读取 JSON 配置文件
    config = read_config_from_json(args.config)

    # 使用配置文件中的参数覆盖命令行参数
    args = override_args_by_config(args, config)

    print(f'args: {args}')

    return args


# 函数用于读取 JSON 配置文件
def read_config_from_json(json_file_path):
    try:
        with open(json_file_path, 'r') as json_file:
            args_dict = json.load(json_file)
        return args_dict
    except FileNotFoundError:
        print(f"Configuration file {json_file_path} not found.")
        return {}

# 函数用于覆盖命令行参数
def override_args_by_config(args, config):
    for key, value in config.items():
        if hasattr(args, key):
            setattr(args, key, value)
    return args




def main():
    args = get_args()
    set_random_seed(args.random_seed)
    initialize_distributed(args)

    if 'UPR' in args.method.upper():
        reranker = UnsupervisedPassageReranker(args)
    elif 'MONOT5' in args.method.upper():
        reranker = MonoT5Reranker(args)
    elif 'BGE' in args.method.upper():
        reranker = BGEReranker(args)
    elif 'RANKGPT' in args.method.upper():
        reranker = RankGPTReranker(args)
    else:
        raise NotImplementedError
    reranker.do_inference()


if __name__ == "__main__":
    main()
