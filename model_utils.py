import torch
import torch.nn.functional as F
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
import torchvision.models.detection as detection_models
from PIL import Image
from breed_info import get_breed_info

# Preserve the original ResNet50 breed classifier
RESNET_WEIGHTS = ResNet50_Weights.DEFAULT
IMAGENET_LABELS = RESNET_WEIGHTS.meta["categories"]

resnet_model = models.resnet50(weights=RESNET_WEIGHTS)
resnet_model.eval()

# COCO Faster R-CNN for dog detection
DETECTION_WEIGHTS = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
detection_model = detection_models.fasterrcnn_resnet50_fpn(weights=DETECTION_WEIGHTS)
detection_model.eval()

# Shared preprocessing for ResNet50
preprocess = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# COCO category id for dog
COCO_DOG_CLASS_ID = 18
# ImageNet class indices for dog breeds
DOG_CLASS_RANGE = range(151, 269)
DEFAULT_DETECTION_THRESHOLD = 0.7


def _ensure_rgb(image: Image.Image) -> Image.Image:
    if image.mode != "RGB":
        return image.convert("RGB")
    return image


def detect_dog_regions(image: Image.Image, detection_threshold: float = DEFAULT_DETECTION_THRESHOLD):
    image = _ensure_rgb(image)
    image_tensor = transforms.ToTensor()(image).unsqueeze(0)

    with torch.no_grad():
        predictions = detection_model(image_tensor)

    if not predictions:
        return []

    output = predictions[0]
    boxes = output.get("boxes", [])
    labels = output.get("labels", [])
    scores = output.get("scores", [])

    dog_regions = []
    for box, label, score in zip(boxes, labels, scores):
        if int(label.item()) == COCO_DOG_CLASS_ID and float(score.item()) >= detection_threshold:
            dog_regions.append((tuple(box.tolist()), float(score.item())))

    return sorted(dog_regions, key=lambda item: item[1], reverse=True)


def crop_regions(image: Image.Image, boxes):
    image = _ensure_rgb(image)
    crops = []
    for box in boxes:
        x1, y1, x2, y2 = [int(max(0, round(v))) for v in box]
        x2 = min(image.width, x2)
        y2 = min(image.height, y2)
        if x2 <= x1 or y2 <= y1:
            crops.append(image.copy())
        else:
            crops.append(image.crop((x1, y1, x2, y2)))
    return crops


def _classify_crop(image: Image.Image):
    image = _ensure_rgb(image)
    input_tensor = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        output = resnet_model(input_tensor)
        probs = F.softmax(output, dim=1)[0]

    predicted_idx = int(torch.argmax(probs))
    breed = IMAGENET_LABELS[predicted_idx]
    metadata = get_breed_info(predicted_idx)
    is_dog = predicted_idx in DOG_CLASS_RANGE

    if is_dog:
        dog_probs = probs[list(DOG_CLASS_RANGE)]
        total_dog_prob = dog_probs.sum().item()
        top_dog_prob = float(probs[predicted_idx].item())
        confidence = top_dog_prob / total_dog_prob if total_dog_prob > 0 else 0.0
    else:
        confidence = float(probs[predicted_idx].item())

    return {
        "is_dog": is_dog,
        "breed": breed,
        "origin": metadata["origin"],
        "lifespan": metadata["lifespan"],
        "confidence": confidence,
        "class_index": predicted_idx,
    }


def predict_dog_breed(image: Image.Image):
    return _classify_crop(image)


def classify_detected_regions(image: Image.Image, detection_threshold: float = DEFAULT_DETECTION_THRESHOLD):
    image = _ensure_rgb(image)
    regions = detect_dog_regions(image, detection_threshold=detection_threshold)

    if not regions:
        return []

    boxes = [region[0] for region in regions]
    scores = [region[1] for region in regions]
    crops = crop_regions(image, boxes)

    results = []
    for idx, crop in enumerate(crops):
        prediction = _classify_crop(crop)
        prediction.update(
            {
                "region_index": idx,
                "bbox": boxes[idx],
                "detection_score": scores[idx],
            }
        )
        results.append(prediction)

    return results


def analyze_image(image: Image.Image, detection_threshold: float = DEFAULT_DETECTION_THRESHOLD):
    image = _ensure_rgb(image)
    detected_regions = classify_detected_regions(image, detection_threshold=detection_threshold)

    if detected_regions:
        return {"detected_regions": detected_regions}

    fallback = predict_dog_breed(image)
    return {"detected_regions": [], "fallback_prediction": fallback}