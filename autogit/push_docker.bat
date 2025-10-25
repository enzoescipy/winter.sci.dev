docker image tag ghcr.io/enzoescipy/winter-sci-dev:pending %1
docker rmi ghcr.io/enzoescipy/winter-sci-dev:pending
docker push %1
