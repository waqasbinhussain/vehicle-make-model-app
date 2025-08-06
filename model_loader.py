from transformers import ViTFeatureExtractor, ViTForImageClassification
from PIL import Image
import torch

# ✅ This model works reliably on Streamlit Cloud
model = ViTForImageClassification.from_pretrained("nateraw/vit-base-patch16-224-inaturalist")
extractor = ViTFeatureExtractor.from_pretrained("nateraw/vit-base-patch16-224-inaturalist")

def predict_vehicle(image: Image.Image):
    inputs = extractor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    label = model.config.id2label[predicted_class_idx]
    confidence = torch.nn.functional.softmax(logits, dim=-1)[0][predicted_class_idx].item()
    return {"label": label, "confidence": round(confidence, 3)}
