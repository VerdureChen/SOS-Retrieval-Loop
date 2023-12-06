from trec_eval import EvalFunction
import argparse

def evaluate_trec(qrel_path, run_path):
    EvalFunction.eval(['-c', '-l', '2', '-m', 'recall.100', qrel_path, run_path])
    EvalFunction.eval(['-c', '-l', '2', '-M', '100', '-m', 'map', qrel_path, run_path])
    EvalFunction.eval(['-c', '-l', '2', '-M', '100', '-m', 'recip_rank', qrel_path, run_path])
    EvalFunction.eval(['-c', '-m', 'ndcg_cut.10', qrel_path, run_path])