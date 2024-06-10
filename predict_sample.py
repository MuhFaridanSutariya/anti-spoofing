import torch
import argparse
from PIL import Image
import matplotlib.pyplot as plt

from src.models.ln_model import ModelInterface
from models.resnext50 import SEResNeXT50
from src.utils import load_transform, load_backbone
import os

def predict_sample(args):
    preprocess = load_transform()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create an instance of your model
    backbone = load_backbone(args)
    model = ModelInterface.load_from_checkpoint(args.model_checkpoint, 
                                            model=backbone,
                                            input_shape=args.input_shape, 
                                            num_classes=args.num_classes)

    model.to(device)
    model.eval()

    if type(args.image) == str:
        image = Image.open(args.image).convert('RGB')

    # Apply the transformations to the input image
    image = preprocess(image).unsqueeze(0)
    image = image.to(device)

    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1)

    # Get the predicted class and the associated probabilities
    predicted_class = torch.argmax(probabilities, dim=1).item()
    predicted_probabilities = probabilities.squeeze().cpu().numpy()

    if predicted_class == 0:
        result = {'tensor': [1, 0], 'class': 0, 'label': 'real', 'probability': predicted_probabilities[0]}
    else:
        result = {'tensor': [0, 1], 'class': 1, 'label': 'fake', 'probability': predicted_probabilities[1]}

    plt.imshow(image.squeeze().permute(1, 2, 0).cpu().numpy())
    plt.title(f"Predict: {result['class']} - {result['label']} - prob: {result['probability']:.4f}")
    plt.axis("off")
    plt.show()

    return result

def main():
    IMAGE_NAME = "2.png"
    MODEL_NAME = "seresnext50.ckpt"
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_checkpoint", type=str, default=os.path.abspath(os.path.join(os.path.dirname(__file__), "checkpoint", MODEL_NAME)))
    parser.add_argument("--image", type=str, default=os.path.abspath(os.path.join(os.path.dirname(__file__), "figures", "samples", IMAGE_NAME)))
    parser.add_argument("--modelname", type=str, default="seresnext50")
    parser.add_argument("--input_shape", type=tuple, default=(3,224,224))
    parser.add_argument("--num_classes", type=int, default=2)
    args = parser.parse_args()

    result = predict_sample(args)
    print(result)

if __name__ == "__main__":
    main()
