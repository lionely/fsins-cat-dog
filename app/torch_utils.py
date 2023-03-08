import io
import torch 
import torch.nn as nn 
import torch.nn.functional as F
import torchvision.transforms as transforms 
from PIL import Image

# load model
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5) # (input channels, )
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 3) # (input prev, output classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = Net()

PATH = "app/cat_dog_net.pth"
model.load_state_dict(torch.load(PATH))
model.eval()

# image -> tensor
def transform_image(image_bytes):
    transform = transforms.Compose([transforms.Grayscale(num_output_channels=3),
                                    transforms.Resize((32,32)), #Model expects 32x32 images
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    image = Image.open(io.BytesIO(image_bytes))
    return transform(image).unsqueeze(0)

# predict
def get_prediction(image_tensor):
    #images = image_tensor.reshape(-1, 32*32)
   #print('image tensor shape',image_tensor.size())
    try:
        outputs = model(image_tensor)
    except Exception as e:  
        print(e)
    #print('model outputted', outputs)
        # max returns (value ,index)
    _, predicted = torch.max(outputs.data, 1)
    return predicted