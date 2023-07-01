#makefile to work with dockerfile

#vars
USERNAME = default
PROJECT = emlop
TAG = assignment6-v1

#setup make commands
help:
	@echo "Makefile supported commands:"
	@echo "build-image: Build image from Dockerfile"
	@echo "run-interactive: Run docker container in interactive mode"
	@echo "run-train: Run training on default config"

build-image:
	docker build -t ${USERNAME}/${PROJECT}:${TAG} . 

run-interactive:
	docker run -it --rm --name ${TAG} \
	-v ${PWD}:/workspace \
	-w /workspace \
	--net=host \
	--shm-size=3gb \
	${USERNAME}/${PROJECT}:${TAG} bash

run-train:
	@echo 'config change ${COMMAND} is defined'
	docker run -it --rm --name ${TAG} \
	-v ${PWD}/outputs:/src/outputs \
	-v ${PWD}/data:/src/data \
	-v ${PWD}/logs:/src/logs \
	${USERNAME}/${PROJECT}:${TAG} \
	copper_train ${COMMAND}

run-multitrain:
	@echo 'num_jobs ${JOB} is defined'
	@echo 'config change ${COMMAND} is defined'
	docker run -it --rm --name ${TAG} \
	-v ${PWD}/outputs:/src/outputs \
	-v ${PWD}/data:/src/data \
	-v ${PWD}/logs:/src/logs \
	--net=host \
	${USERNAME}/${PROJECT}:${TAG} \
	copper_train -m hydra/launcher=joblib hydra.launcher.n_jobs=${JOB} ${COMMAND}

run-visualize:
	@echo 'starting AIM and MLflow UI'
	docker exec -d ${TAG} aim up
	docker exec -d ${TAG} bash -c "cd logs/mlflow && mlflow ui"
	@echo 'AIM UI is running in localhost:43800'
	@echo 'MLflow UI is running in localhost:5000'

run-eval:
	@echo 'config change ${COMMAND} is defined'
	docker run -it --rm --name ${TAG} \
	-v ${PWD}/outputs:/src/outputs \
	-v ${PWD}/data:/src/data \
	-v ${PWD}/logs:/src/logs \
	${USERNAME}/${PROJECT}:${TAG} \
	copper_eval ${COMMAND}

push-image:
	echo "docker push ${USERNAME}/${PROJECT}:${TAG}" 

