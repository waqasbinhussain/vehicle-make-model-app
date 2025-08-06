from transformers import ViTForImageClassification, ViTFeatureExtractor
from PIL import Image
import torch

model = ViTForImageClassification.from_pretrained("hyperconnect/vmmr-model")
extractor = ViTFeatureExtractor.from_pretrained("hyperconnect/vmmr-model")

def predict_vehicle(image: Image.Image):
    inputs = extractor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    label = model.config.id2label[predicted_class_idx]
    return {"label": label, "confidence": round(torch.nn.functional.softmax(logits, dim=-1)[0][predicted_class_idx].item(), 3)}
