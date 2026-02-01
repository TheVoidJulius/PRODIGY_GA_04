import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from utils.image_loader import load_image, save_image
from utils.losses import gram_matrix
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"


content = load_image("images/content/content.jpg", 256)
style = load_image("images/style/style.jpg", 256)
generated = content.clone().requires_grad_(True)


vgg = models.vgg19(pretrained=True).features.to(device).eval()


content_layer = "21"
style_layers = ["0", "5", "10", "19", "28"]

content_weight = 1
style_weight = 1e6

optimizer = optim.Adam([generated], lr=0.02)

for step in tqdm(range(150)):
    gen_features = {}
    content_features = {}
    style_features = {}

    x = generated
    y = content
    z = style

    for name, layer in vgg._modules.items():
        x = layer(x)
        y = layer(y)
        z = layer(z)

        if name == content_layer:
            content_features[name] = y
            gen_features[name] = x

        if name in style_layers:
            style_features[name] = z
            gen_features[name] = x


    content_loss = torch.mean((gen_features[content_layer] - content_features[content_layer]) ** 2)


    style_loss = 0
    for layer in style_layers:
        gen_gram = gram_matrix(gen_features[layer])
        style_gram = gram_matrix(style_features[layer])
        style_loss += torch.mean((gen_gram - style_gram) ** 2)

    total_loss = content_weight * content_loss + style_weight * style_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if step % 50 == 0:
        print(f"Step {step} | Loss: {total_loss.item()}")

save_image(generated, "images/output/result.png")
print("✅ Style transfer complete!")
