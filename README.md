# EMLO-3_Assignment8
## Introduction
This repo contains session-8 assignment of EMLO course from TSAI.
- Add Demo Deployment for your trained ViT model (scripted checkpoint)
- Convert your model to TorchScript SCRIPTED Model
Model should be trained on CIFAR10 Dataset
Handle Image Resizing to your model input size
Conversion and saving must happen after model training is done
Demo must accept image from user, and give the top 10 predictions

- Demo MUST be integrated into the pytorch-lightning template
Dockerize the Demo
- package the model (scripted only) inside the docker
the docker image must be CPU only
- docker image size limit is 1.5GB
- docker run <image>:<tag> should start the webapp !
use port 8080 for the webserver

- Demo MUST be integrated into the pytorch-lightning template
Dockerize the Demo
- package the model (scripted only) inside the docker
the docker image must be CPU only
- docker image size limit is 1.5GB

## Getting started
#### Training 
- Train the model using docker by using command `make run-train COMMAND="experiment=cifar tuner=False train=True"`
- Train the model without docker by using command:
    - `pip install -r requirements.txt && pip install -e .`
    - `copper_train experiment=cifar tuner=False train=True`"
- To use gpu while training add following argument `trainer.accelerator=gpu`
#### Inference
- Run gradio app using hydra and template by this command `copper_infer demo_ckpt_path=/path/t0/scripted_model` 
- Run gradio app as stand-alone docker app by this command:
    - `docker build -f gradio_Dockerfile -t gradio_infer:v1 .`
    - `docker run -it -p 8080:8180 gradio_infer`
- **Built image can also be pulled directly from dockerhub** `docker pull devdastl/emlop:assignment8-v1-infer`
- Run pulled image using make `export USERNAME=devdastl && make run-infer`


