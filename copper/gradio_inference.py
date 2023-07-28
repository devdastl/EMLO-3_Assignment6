from typing import Tuple
import os
import torch
import hydra
import gradio as gr
from omegaconf import DictConfig
from torchvision import transforms
import torch.nn.functional as F

from copper import utils

log = utils.get_pylogger(__name__)

os.environ["GRADIO_SERVER_PORT"] = "8080"

# Define a transformation to resize images to 32x32
resize_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

class_labels = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

def classify_image(model, image):
    if image is None:
        return None

    # Apply the resizing transformation
    image = resize_transform(image)
    image = image.unsqueeze(0)

    preds = model.forward(image)
    preds = F.softmax(preds, dim=1)
    preds = preds[0].tolist()
    # Map the class labels to the predictions
    labeled_preds = {class_labels[i]: preds[i] for i in range(10)}
    return labeled_preds

def demo(cfg: DictConfig) -> Tuple[dict, dict]:


    assert cfg.demo_ckpt_path

    log.info("Running Demo")

    log.info(f"Instantiating scripted model <{cfg.demo_ckpt_path}>")
    model = torch.jit.load(cfg.demo_ckpt_path)

    log.info(f"Loaded Model: {model}")

    im = gr.Image(shape=(32, 32), image_mode="RGB", source="upload")

    demo = gr.Interface(
        fn=lambda image: classify_image(model, image),
        inputs=[im],
        outputs=[gr.Label(num_top_classes=10)],
        live=True,
    )

    demo.launch(server_name="0.0.0.0", server_port=8080)

@hydra.main(
    version_base="1.2", config_path="../configs", config_name="demo.yaml"
)
def main(cfg: DictConfig) -> None:
    demo(cfg)

if __name__ == "__main__":
    main()