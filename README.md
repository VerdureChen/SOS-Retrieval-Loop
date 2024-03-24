# Codebase of "Spiral of Silence: How is Large Language Model Killing Information Retrieval?—A Case Study on Open Domain Question Answering"
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#news-and-updates">News and Updates</a></li>
    <li><a href="#introduction">Introduction</a></li>
    <li><a href="#installation">Installation</a></li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#supported-datasets-and-models">Datasets and Models</a></li>
    <li><a href="#benchmark-results">Results</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

<!-- News and Updates -->

## News and Updates
- [05/12/2024] Published code used in our experiments.


<!-- Introduction -->

## Introduction

In this study, we construct and iteratively run a simulation pipeline to deeply investigate the short-term and long-term effects of LLM text on RAG systems.

![Pipeline Structure](pipeline.png)



### What does our code currently provide?
<!--
1. 方便使用的迭代模拟工具：我们提供了一个易于使用的迭代模拟工具，通过融合ElasticSearch、LangChain和api-for-llm的功能，能够方便地加载数据集，选择各种LLM和检索排序模型和自动迭代模拟。
2. 支持多个数据集：包括但不限于Natural Questions, TriviaQA, WebQuestions, PopQA，通过将数据转化为jsonl格式，你可以使用本文的框架对任何数据进行实验。
3. 支持多种检索模型和重排序模型：BM25, Contriever，LLM-Embedder，BGE，UPR，MonoT5等。
4. 支持多种RAG pipeline演变评估方法，对每次实验的大量结果进行自动评价整理。
-->

1. **User-friendly Iteration Simulation Tool:** We offer an easy-to-use iteration simulation tool that integrates functionalities from [ElasticSearch](https://www.elastic.co/elasticsearch/), [LangChain](https://github.com/langchain-ai/langchain/), and [api-for-open-llm](https://github.com/xusenlinzy/api-for-open-llm), allowing for convenient dataset loading, selection of various LLMs and retrieval-ranking models, and automated iterative simulation.
2. **Support for Multiple Datasets:** Including but not limited to [Natural Questions](https://github.com/google-research-datasets/natural-questions), [TriviaQA](https://github.com/mandarjoshi90/triviaqa), [WebQuestions](https://github.com/brmson/dataset-factoid-webquestions), [PopQA](https://github.com/AlexTMallen/adaptive-retrieval). By converting data to jsonl format, you can use the framework in this paper to experiment with any data.
3. **Support for Various Retrieval and Re-ranking Models:** [BM25](https://python.langchain.com/docs/integrations/retrievers/elastic_search_bm25), [Contriever](https://github.com/facebookresearch/contriever), [LLM-Embedder](https://github.com/FlagOpen/FlagEmbedding), [BGE](https://github.com/FlagOpen/FlagEmbedding), [UPR](https://github.com/DevSinghSachan/unsupervised-passage-reranking), [MonoT5](https://github.com/castorini/pygaggle) and more.
4. **Supports Various RAG Pipeline Evolution Evaluation Methods:** Automatically organizes and assesses the vast amount of results from each experiment.


<!-- GETTING STARTED -->

## Installation
<!-- 
我们的框架依赖ElasticSearch 8.11.1和api-for-open-llm，因此需要先安装这两个工具。我们建议您下载相同版本的[ElasticSearch 8.11.1](https://www.elastic.co/guide/en/elasticsearch/reference/8.11/targz.html)，并且在启动前从其config/elasticsearch.yml文件中设置好对应的http.port和http.host，它们将用于本仓库代码运行的配置。
当您安装api-for-open-llm时，请根据其[官方仓库](https://github.com/xusenlinzy/api-for-open-llm/blob/master/docs/SCRIPT.md)的指引，安装好运行您所需模型的依赖和环境。您在.env文件中设置的PORT也将作为本代码库需要的配置。
-->
Our framework depends on **ElasticSearch 8.11.1** and **api-for-open-llm**, therefore it is necessary to install these two tools first. We suggest downloading the same version of [ElasticSearch 8.11.1](https://www.elastic.co/guide/en/elasticsearch/reference/8.11/targz.html), and before starting, set the appropriate `http.port` and `http.host` in the `config/elasticsearch.yml` file, as these will be used for the configuration needed to run the code in this repository.

When installing api-for-open-llm, please follow the instructions provided by its [repository](https://github.com/xusenlinzy/api-for-open-llm/blob/master/docs/SCRIPT.md) to install all the dependencies and environment required to run the model you need. The `PORT` you configure in the `.env` file will also serve as a required configuration for this codebase.

### Install via GitHub

First, clone the repo:
```sh
git clone --recurse-submodules git@github.com:VerdureChen/SOS-Retrieval-Loop.git
```

Then, 

```sh
cd SOS-Retrieval-Loop
```

To install the required packages, you can create a conda environment:

```sh
conda create --name SOS_LOOP python=3.10
```

then use pip to install required packages:

```sh
pip install -r requirements.txt
```

## Usage

Please see [Installation](#installation) to install the required packages.
<!-- 
在运行我们的框架之前，您需要先启动ElasticSearch和api-for-open-llm。在启动ElasticSearch时，您需要在其config/elasticsearch.yml文件中设置好对应的http.port和http.host，它们将用于本仓库代码运行的配置。
在启动api-for-open-llm时，您需要在.env文件中设置好PORT，它也将作为本代码库需要的配置。
-->
Before running our framework, you need to start **ElasticSearch** and **api-for-open-llm**. When starting **ElasticSearch**, you need to set the appropriate `http.port` and `http.host` in the `config/elasticsearch.yml` file, as these will be used for the configuration needed to run the code in this repository.

When starting **api-for-open-llm**, you need to set the `PORT` in the `.env` file, which will also serve as a required configuration for this codebase.

<!--
### Configuration
由于我们的代码涉及到较多的数据集、模型和索引等功能，我们使用三级config方式来控制代码的运行：
1. 在代码的特定功能（如检索、重排、生成等）目录下的config文件夹内，存放着该功能的配置文件模板，其中包含了所有可自行配置的参数，你可以从配置文件的文件名中确认该配置文件对应功能和数据集。例如，`src/retrieval_loop/retrieve_configs/bge-base-config-nq.json`是一个检索配置文件，对应的数据集是Natural Questions，使用的模型是BGE-base。我们以`src/retrieval_loop/index_configs/bge-base-config-psgs_w100.json`为例：
```json
{
      "new_text_file": "../../data_v2/input_data/DPR/psgs_w100.jsonl", 
      "retrieval_model": "bge-base",
      "index_name": "bge-base_faiss_index",
      "index_path": "../../data_v2/indexes",
      "index_add_path": "../../data_v2/indexes",
      "page_content_column": "contents",
      "index_exists": false,
      "normalize_embeddings": true,
      "query_files": ["../../data_v2/input_data/DPR/nq-test-h10.jsonl"],
      "query_page_content_column": "question",
      "output_files": ["../../data_v2/ret_output/DPR/nq-test-h10-bge-base"],
      "elasticsearch_url": "http://124.16.138.142:9978"
}
```
其中，`new_text_file`是需要新添加到索引的文档路径，`retrieval_model`是使用的检索模型，`index_name`是索引的名称，`index_path`是索引的存储路径，`index_add_path`是索引的增量文档在索引中的ID存放路径（这对于我们需要从索引中删除特定文档特别有用），`page_content_column`是文档文件中待索引的文本的列名，`index_exists`指示索引是否已经存在（如果设置为false则会新建相应索引，否则从路径读取已存在的索引），`normalize_embeddings`是是否对检索模型的输出进行归一化，`query_files`是查询文件的路径，`query_page_content_column`是查询文件中查询文本的列名，`output_files`是输出检索结果文件的路径，`elasticsearch_url`是ElasticSearch的url。
2. 由于我们通常需要在一个pipeline中融合多个步骤，与示例脚本`src/run_loop.sh`相适应的，我们提供一个全局配置文件`src/test_function/test_configs/template_total_config.json`，在这个全局配置文件中，你可以一次性对每个阶段的参数进行配置，你不必在其中配置所有参数，只需配置你相对于模板配置文件需要修改的参数即可。
3. 为了提升脚本运行效率，`src/run_loop.sh`脚本支持对于某个检索-重排方法，同时运行多个数据集和LLM生成的结果，为了能够更灵活地配置此类实验，我们支持在`src/run_loop.sh`内部通过`rewrite_configs.py`在pipeline运行过程中生成新的配置文件。例如，当我们需要循环运行pipeline时，我们需要记录每一轮的config内容，在每次进行检索之前，脚本将运行：
```bash
python ../rewrite_configs.py --total_config "${USER_CONFIG_PATH}" \
                          --method "${RETRIEVAL_MODEL_NAME}" \
                          --data_name "nq" \
                          --loop "${LOOP_NUM}" \
                          --stage "retrieval" \
                          --output_dir "${CONFIG_PATH}" \
                          --overrides '{"query_files": ['"${QUERY_FILE_LIST}"'], "output_files": ['"${OUTPUT_FILE_LIST}"'] , "elasticsearch_url": "'"${elasticsearch_url}"'", "normalize_embeddings": false}'
```
其中，`--total_config`是全局配置文件的路径，`--method`是检索方法的名称，`--data_name`是数据集的名称，`--loop`是当前循环的轮数，`--stage`是当前pipeline的阶段，`--output_dir`是新生成的配置文件的存储路径，`--overrides`是需要修改的参数（同样是每个任务config模板的子集）。
4. 在进行配置时你需要注意三类配置的优先级：第一级是每个任务config模板中的默认配置，第二级是全局配置文件，第三级是pipeline运行过程中生成的新配置文件。在pipeline运行过程中，第二级配置会覆盖第一级配置，第三级配置会覆盖第二级配置。
-->

### Configuration
Since our code involves many datasets, models, and index functionalities, we use a three-level config method to control the operation of the code:
1. In the config folder of the specific function (such as retrieval, re-ranking, generation, etc.), there is a configuration file template for that function, which contains all the parameters that can be configured by yourself. You can confirm the function and dataset corresponding to the configuration file from the file name of the configuration file. For example, `src/retrieval_loop/retrieve_configs/bge-base-config-nq.json` is a retrieval configuration file, corresponding to the Natural Questions dataset, and using the BGE-base model. We take `src/retrieval_loop/index_configs/bge-base-config-psgs_w100.json` as an example:
    ```json
    {
          "new_text_file": "../../data_v2/input_data/DPR/psgs_w100.jsonl", 
          "retrieval_model": "bge-base",
          "index_name": "bge-base_faiss_index",
          "index_path": "../../data_v2/indexes",
          "index_add_path": "../../data_v2/indexes",
          "page_content_column": "contents",
          "index_exists": false,
          "normalize_embeddings": true,
          "query_files": ["../../data_v2/input_data/DPR/nq-test-h10.jsonl"],
          "query_page_content_column": "question",
          "output_files": ["../../data_v2/ret_output/DPR/nq-test-h10-bge-base"],
          "elasticsearch_url": "http://124.16.138.142:9978"
    }
    ```
    Where `new_text_file` is the path to the document to be newly added to the index, `retrieval_model` is the retrieval model used, `index_name` is the name of the index, `index_path` is the storage path of the index, `index_add_path` is the path where the ID of the incremental document in the index is stored (this is particularly useful when we need to delete specific documents from the index), `page_content_column` is the column name of the text to be indexed in the document file, `index_exists` indicates whether the index already exists (if set to false, the corresponding index will be created, otherwise the existing index will be read from the path), `normalize_embeddings` is whether to normalize the output of the retrieval model, `query_files` is the path to the query file, `query_page_content_column` is the column name of the query text in the query file, `output_files` is the path to the output retrieval result file, and `elasticsearch_url` is the url of ElasticSearch.
2. Since we usually need to integrate multiple steps in a pipeline, corresponding to the example script `src/run_loop.sh`, we provide a global configuration file `src/test_function/test_configs/template_total_config.json`. In this global configuration file, you can configure the parameters of each stage at once. You do not need to configure all the parameters in it, just the parameters you need to modify relative to the template configuration file.
3. In order to improve the efficiency of script running, the `src/run_loop.sh` script supports running multiple datasets and LLM-generated results for a retrieval-re-ranking method at the same time. To flexibly configure such experiments, we support generating new configuration files during the pipeline run in `src/run_loop.sh` through `rewrite_configs.py`. For example, when we need to run the pipeline in a loop, we need to record the config content of each round. Before each retrieval, the script will run:
    ```bash
    python ../rewrite_configs.py --total_config "${USER_CONFIG_PATH}" \
                              --method "${RETRIEVAL_MODEL_NAME}" \
                              --data_name "nq" \
                              --loop "${LOOP_NUM}" \
                              --stage "retrieval" \
                              --output_dir "${CONFIG_PATH}" \
                              --overrides '{"query_files": ['"${QUERY_FILE_LIST}"'], "output_files": ['"${OUTPUT_FILE_LIST}"'] , "elasticsearch_url": "'"${elasticsearch_url}"'", "normalize_embeddings": false}'
    ```
    Where `--total_config` is the path to the global configuration file, `--method` is the name of the retrieval method, `--data_name` is the name of the dataset, `--loop` is the number of the current loop, `--stage` is the stage of the current pipeline, `--output_dir` is the storage path of the newly generated configuration file, and `--overrides` is the parameters that need to be modified (also a subset of the template configuration file for each task).
4. When configuring, you need to pay attention to the priority of the three types of configurations: the first level is the default configuration in each task config template, the second level is the global configuration file, and the third level is the new configuration file generated during the pipeline run. During the pipeline run, the second level configuration will override the first level configuration, and the third level configuration will override the second level configuration.


<!--
### Running the Code
通过以下步骤，你可以复现我们的实验：
1. 数据集预处理：不论是查询还是文档，我们都需要将数据集转化为jsonl格式。我们的实验中使用data.wikipedia_split.psgs_w100数据，可参考[DPR仓库](https://github.com/facebookresearch/DPR?tab=readme-ov-file#resources--data-formats)的说明将其下载至`data_v2/raw_data/DPR`目录下并解压。我们提供一个简单的脚本`data_v2/gen_dpr_hc_jsonl.py`，可以将数据集转化为jsonl格式并放置于`data_v2/input_data/DPR`。实验中使用到的query文件位于`data_v2/input_data/DPR/sampled_query`。
2. 生成Zero-shot RAG结果：使用`src/llm_zero_generate/run_generate.sh`，通过修改文件中的配置，能够批量化生成所有数据和模型的zero-shot RAG结果。
-->

