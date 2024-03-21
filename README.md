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

![Pipeline Structure](.pipeline.png)



### What does our code currently provide?
<!--
1. 方便使用的迭代模拟工具：我们提供了一个易于使用的迭代模拟工具，通过融合ElasticSearch、LangChain和api-for-llm的功能，能够方便地加载数据集，选择各种LLM和检索排序模型和自动迭代模拟。
2. 支持多个数据集：包括但不限于Natural Questions, TriviaQA, WebQuestions, PopQA，通过将数据转化为jsonl格式，你可以使用本文的框架对任何数据进行实验。
3. 支持多种检索模型和重排序模型：BM25, Contriever，LLM-Embedder，BGE，UPR，MonoT5等。
4. 支持多种RAG pipeline演变评估方法，对每次实验的大量结果进行自动评价整理。
-->

1. **User-friendly Iteration Simulation Tool:** We offer an easy-to-use iteration simulation tool that integrates functionalities from [ElasticSearch](https://www.elastic.co/elasticsearch/), [LangChain](https://github.com/langchain-ai/langchain/), and [api-for-open-llm](https://github.com/xusenlinzy/api-for-open-llm), allowing for convenient dataset loading, selection of various LLMs and retrieval-ranking models, and automated iterative simulation.
2. **Support for Multiple Datasets:** Including but not limited to [Natural Questions](https://github.com/google-research-datasets/natural-questions), [TriviaQA](https://github.com/mandarjoshi90/triviaqa), [WebQuestions](https://github.com/brmson/dataset-factoid-webquestions), [PopQA](https://github.com/AlexTMallen/adaptive-retrieval). By converting data to jsonl format, you can use the framework in this paper to experiment with any data.
3. **Support for Various Retrieval and Re-ranking Models:** [BM25](https://), [Contriever](https://), [LLM-Embedder](https://), [BGE](https://), [UPR](https://), [MonoT5](https://) and more.
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
