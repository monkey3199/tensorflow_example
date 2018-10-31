TAG := $(shell date +v%Y%m%d)
IMG = tensorflow-examples/mnist_cnn_gpu

all: run

# To build without the cache set the environment variable
# export DOCKER_BUILD_OPTS=--no-cache
build:
	docker build ${DOCKER_BUILD_OPTS} -t $(IMG):$(TAG) . 
	docker tag $(IMG):$(TAG) $(IMG):latest
	@echo Built $(IMG):latest
	@echo Built $(IMG):$(TAG)

run:
	docker run ${DOCKER_BUILD_OPTS} -it $(IMG):latest . 
	@echo Run $(IMG):latest


# Build but don't attach the latest tag. This allows manual testing/inspection of the image
# first.