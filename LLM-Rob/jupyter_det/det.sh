#wget http://git.cipsup.cn/snippets/28/raw\?inline\=false -O determined.Dockerfile

#BASE_IMAGE=docker.cipsup.cn/chenxiaoyang/reranker_full
BASE_IMAGE=chenxiaoyang/llm-api
docker build --rm                   \
    -f Dockerfile                   \
    -t ${BASE_IMAGE}-det0.19.10 \
    --build-arg BASE_IMAGE=${BASE_IMAGE} \
    --build-arg DET_VERSION=0.19.10 \
    --build-arg http_proxy=http://192.168.14.70:7890 \
    --build-arg https_proxy=http://192.168.14.70:7890 \
    "."
