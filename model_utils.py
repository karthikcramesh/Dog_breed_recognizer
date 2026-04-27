import torch
import torch.nn.functional as F
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from PIL import Image
from breed_info import get_breed_info

# Load the pre-trained ResNet50 model with updated API
weights = ResNet50_Weights.DEFAULT
labels = weights.meta["categories"]

model = models.resnet50(weights=weights)
model.eval()

# Define the preprocessing transforms
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ImageNet class indices for dogs (approximate range: 151-268)
DOG_CLASS_RANGE = range(151, 269)

def predict_dog_breed(image):
    """Predict if the image contains a dog and return breed details."""
    # Convert any non-RGB image to RGB so preprocessing works correctly.
    if image.mode != 'RGB':
        image = image.convert('RGB')

    input_tensor = preprocess(image).unsqueeze(0)
    
    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1)[0]
    
    predicted_idx = int(torch.argmax(probs))
    confidence = float(probs[predicted_idx].item())
    breed = labels[predicted_idx]
    metadata = get_breed_info(predicted_idx)
    
    return {
        "is_dog": predicted_idx in DOG_CLASS_RANGE,
        "breed": breed,
        "origin": metadata["origin"],
        "lifespan": metadata["lifespan"],
        "confidence": confidence,
        "class_index": predicted_idx,
    }