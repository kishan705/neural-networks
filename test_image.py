from nl import ResNet
import torch
from PIL import Image
from torchvision import transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

# Rest of your code

# ✅ Load your trained ResNet model
model = ResNet(num_classes=10).to(device)
model.load_state_dict(torch.load('resnet_cifar10.pth'))
model.eval()

# ✅ CIFAR-10 classes
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

# ✅ Load and preprocess your image
image_path = '/Users/kishanamaliya/Downloads/girl.webp'
image = Image.open(image_path).convert('RGB')

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

image = transform(image).unsqueeze(0).to(device)

# ✅ Run inference
with torch.no_grad():
    output = model(image)
    predicted_class = output.argmax(dim=1).item()

# ✅ Print the result
print(f"Predicted class: {classes[predicted_class]}")
