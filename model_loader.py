from transformers import AutoProcessor, AutoModelForObjectDetection
from PIL import Image
import torch

# Load model and processor
processor = AutoProcessor.from_pretrained("keremberke/yolov5m-v7-vehicle-make-model")
model = AutoModelForObjectDetection.from_pretrained("keremberke/yolov5m-v7-vehicle-make-model")

def predict_vehicle(image: Image.Image):
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, threshold=0.5, target_sizes=target_sizes)[0]

    predictions = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        predictions.append({
            "label": model.config.id2label[label.item()],
            "score": round(score.item(), 2),
            "box": [round(b, 2) for b in box.tolist()]
        })
    return predictions
