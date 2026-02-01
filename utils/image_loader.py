import torch
from PIL import Image
import torchvision.transforms as transforms

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_image(path, size=512):
    image = Image.open(path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor()
    ])
    image = transform(image).unsqueeze(0)
    return image.to(device)

def save_image(tensor, path):
    image = tensor.clone().detach().cpu().squeeze(0)
    transforms.ToPILImage()(image).save(path)
