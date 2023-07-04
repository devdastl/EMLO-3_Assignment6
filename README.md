# EMLO-3_Assignment4
This repo contains session-4 assignment of EMLO course from TSAI.

## Introduction
This assignment contains a template containing pytorch lightning code. Below are few points this template covers:

- Dockerfile for the project
    - DevContainer for the Project
    - Docker Image should have your package installed
- Add CIFAR10 Dataset
- Add TIMM to the project template
    - Should be able to train any image classification model from TIMM
## Folder structure
We can get started with the folder structure for this repository.

## Getting started
Lets go through the steps required to run this template:

Prerequisite:

- Docker installed on the host system.
- VS Code if using devcontainer.
- make tool installed on the host.

Steps: 
- Git clone this repo onto the host: `git clone https://github.com/devdastl/EMLO-3_Assignment4.git`
- Build docker image, this repo uses Makefile to easily execute docker command. Run command `make build-image`. you can pass `USERNAME=a PROJECT=b TAG=c` with make command to build docker image of name `a/b:c`.
- You can also directly pull docker image from Dockerhub instead of building it.
    - `docker pull devdastl/emlop:assignment4-v1`
    - `make USERNAME=devdastl run-train`
- Run `make help` to list available steps.
<br>

There are three ways to exectue training and evaluation on Docker container and these are coverd below:

### Run training and evaluation using Makefile
To directly run training and evaluation on built image you can run following make command:
- To start training on CIFAR: `make run-train`
- To train on MNSIT: `make run-train COMMAND="data=mnist model=mnist"`
    - This will train and put model and log in `/outputs/date/time/lightning_logs` as well as a temp model in `/outputs`.
    - 
- To start evaluation: `make run-eval`
- Start eval on MNIST: `make run-eval COMMAND="data=mnist model=mnist"` 

### Run interactive using Makefile

### Run VS code specific devcontainer



```
copper_train --help
```

examples

- `copper_train data.num_workers=16`
- `copper_train data.num_workers=16 trainer.deterministic=True +trainer.fast_dev_run=True`

## Development

Install in dev mode

```
pip install -e .
```

### Docker

<docker-usage-instructions-here>
