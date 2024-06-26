.DEFAULT_GOAL := help # Sets default action to be help


define PRINT_HELP_PYSCRIPT # start of Python section
import re, sys

output = []
# Loop through the lines in this file
for line in sys.stdin:
    # if the line has a command and a comment start with
    #   two pound signs, add it to the output
    match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
    if match:
        target, help = match.groups()
        output.append("%-10s %s" % (target, help))
# Sort the output in alphanumeric order
output.sort()
# Print the help result
print('\n'.join(output))
endef
export PRINT_HELP_PYSCRIPT # End of python section


# Set the Docker image name and tag
IMAGE_NAME := breakout
IMAGE_TAG := latest
SAVE_NAME := breakout.tar 

# Set the Docker container name
CONTAINER_NAME := breakout

# Set any environment variables needed for the container
ENV_VARS := -e ENV_VAR_1=value1 -e ENV_VAR_2=value2

# Set the Docker build context (the directory containing the Dockerfile)
BUILD_CONTEXT := .
 
# Set any additional flags for the Docker build command
BUILD_FLAGS :=

# Set any additional flags for the Docker run command
RUN_FLAGS := -p 8080:80 -v /home/mat/repos/breakout:/src -v /home/mat/data:/data
#GPU_RUN_FLAGS := -p 8080:80 -v /home/ubuntu/repos/breakout:/src -v /home/ubuntu/_data:/data -v /home/ubuntu/data1:/cloud_data
GPU_RUN_FLAGS := -p 8080:80 -v /home/ubuntu/repos/breakout:/src -v /home/ubuntu/_data:/data

help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

prepare: ## Prepare lambda box for use
	sudo usermod -aG docker $(USER)
	newgrp docker
	mkdir /home/ubuntu/_data

## Docker commands
build: ## Build docker repo
	docker build $(BUILD_FLAGS) -t $(IMAGE_NAME):$(IMAGE_TAG) $(BUILD_CONTEXT)

run-local: ## Target to run the Docker container locally to test
	docker run --rm -it $(ENV_VARS) $(RUN_FLAGS) --name $(CONTAINER_NAME) $(IMAGE_NAME):$(IMAGE_TAG)

run-gpu: ## Target to run the Docker container on GPU
	docker run --rm -it --gpus all $(ENV_VARS) $(GPU_RUN_FLAGS) --name $(CONTAINER_NAME) $(IMAGE_NAME):$(IMAGE_TAG)

stop: ## Target to stop and remove the Docker container
	docker stop $(CONTAINER_NAME) || true
	docker rm $(CONTAINER_NAME) || true


clean: stop ## Target to remove the Docker image
	docker rmi $(IMAGE_NAME):$(IMAGE_TAG)

save: ## Save the image locally
	docker save --output ${SAVE_NAME} $(IMAGE_NAME):$(IMAGE_TAG)

load: ## Load the local image
	docker load --input ${SAVE_NAME}


install: ## Install the breakout repo. Run from the src directory
	pip3 install -e .


# Called inside container
train: ## Train model
	train --system.work_dir="/data/experiments" --video_path="/data/videos"


.PHONY: help build run stop clean save load
