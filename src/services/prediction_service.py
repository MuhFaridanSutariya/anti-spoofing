import torch
from PIL import Image
from src.models.ln_model import ModelInterface
from src.utils import load_transform, load_backbone

class Args:
    def __init__(self, modelname, input_shape, num_classes):
        self.modelname = modelname
        self.input_shape = input_shape
        self.num_classes = num_classes

class PredictionService:
    def __init__(self, model_checkpoint, input_shape, num_classes):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.preprocess = load_transform()
        self.model = self.load_model(model_checkpoint, input_shape, num_classes)

    def load_model(self, model_checkpoint, input_shape, num_classes):
        args = Args(modelname="seresnext50", input_shape=input_shape, num_classes=num_classes)
        backbone = load_backbone(args)
        model = ModelInterface.load_from_checkpoint(model_checkpoint, 
                                                    model=backbone,
                                                    input_shape=input_shape, 
                                                    num_classes=num_classes,
                                                    strict=False)
        model.to(self.device)
        model.eval()
        return model

    def predict(self, image_path):
        image = Image.open(image_path).convert('RGB')
        image = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(image)
            probabilities = torch.softmax(outputs, dim=1)

        predicted_class = torch.argmax(probabilities, dim=1).item()
        predicted_probabilities = probabilities.squeeze().cpu().numpy()

        if predicted_class == 0:
            result = {'tensor': [1, 0], 'class': 0, 'label': 'real', 'probability': float(predicted_probabilities[0])}
        else:
            result = {'tensor': [0, 1], 'class': 1, 'label': 'fake', 'probability': float(predicted_probabilities[1])}

        return result
