from typing import Tuple
import sys
import os
import torch
import gradio as gr
from torchvision import transforms
import torch.nn.functional as F
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Define a transformation to resize images to 32x32
resize_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

class_labels = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

def recognize_image(model, image):
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

def demo(demo_ckpt_path: str) -> Tuple[dict, dict]:

    assert demo_ckpt_path

    log.info("Running Demo")

    log.info(f"Instantiating scripted model <{demo_ckpt_path}>")
    model = torch.jit.load(demo_ckpt_path)

    log.info(f"Loaded Model: {model}")

    im = gr.Image(shape=(32, 32), image_mode="RGB", source="upload")

    demo = gr.Interface(
        fn=lambda image: recognize_image(model, image),
        inputs=[im],
        outputs=[gr.Label(num_top_classes=10)],
        live=True,
    )

    demo.launch(server_name="0.0.0.0", server_port=8180)

def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python3 demo_sripted.py <demo_ckpt_path>")
        return

    ckpt_path = sys.argv[1]
    demo(ckpt_path)

if __name__ == "__main__":
    main()