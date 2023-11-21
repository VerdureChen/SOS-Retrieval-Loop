user_name="chenxiaoyang"
project_name="llm-api"
docker_registry="docker.cipsup.cn"

echo | docker login $docker_registry &>/dev/null || docker login $docker_registry -u $user_name
#docker build --no-cache --rm -t $user_name/$project_name .
#DOCKER_BUILDKIT=1
docker build --rm -f "Dockerfile" -t $user_name/$project_name --build-arg http_proxy=http://192.168.14.70:7890 --build-arg https_proxy=http://192.168.14.70:7890 .
docker tag $user_name/$project_name $docker_registry/$user_name/$project_name
docker push $docker_registry/$user_name/$project_name