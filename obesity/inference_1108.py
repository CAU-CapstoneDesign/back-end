"""

Library Import 

"""

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
from PIL import Image
import pandas as pd
import torch.nn as nn
import time

"""

Model Define

input : batch x 3 x seq x 64 x 64
output : num_classes ( 3 )


"""

class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNet3D(nn.Module):
    def __init__(self, num_classes=1):
        super(ResNet3D, self).__init__()
        self.conv1 = nn.Conv3d(3, 64, kernel_size=7, stride=(1, 2, 2), padding=(3, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        layers = []
        strides = [stride] + [1] * (num_blocks - 1)
        for stride in strides:
            layers.append(ResidualBlock3D(in_channels, out_channels, stride))
            in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

"""

Inference Method 

you may use this method in your code.

input
model_path : path of model weight format ( .pt or .pth )
image_paths : 2d list of image paths ( [[ "A_01.jpg ","A_02.jpg ","A_03.jpg ","A_08.jpg "   ][ "B_01.jpg ","B_02.jpg ","B_03.jpg ","B_08.jpg "   ]] )

output 
predictions = [[ A probability of low , A probability of normal , A probability of over   ] , [ B probability of low ,  B probability of normal ,B  probability of over   ] ...] 



"""


def inference(model_path, image_paths):

    transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ResNet3D(num_classes=3).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    predictions = []

    with torch.no_grad():
        for image_path in image_paths:
            imagestack = []
            for angle_image_path in image_path :

                image = Image.open(angle_image_path)
                image = transform(image).unsqueeze(0).to(device)
                imagestack.append(image)
            imagestack = torch.stack(imagestack, dim=2)
            logits = model(imagestack)
            probabilities = F.softmax(logits, dim=1)
            #predicted_classes = torch.argmax(test_probabilities, dim=1).cpu().numpy()
            predictions.extend(probabilities.tolist())
    return predictions[0]



"""

Main Method

you can try inference method.
In this example, image_paths is assigned from the csv file. 
And output it as a percentage ( % ) 

"""

if __name__ == "__main__": 
    model_path = "./checkpoints/data1105POMBody1238_resnet183D_BCS3_2.pt"
    #test_path = "./data/data/test1102.csv" # for ai test. you can ignore this line
    #test_df = pd.read_csv(test_path) # for ai test.  you can ignore this line
    image_paths = [["./data/data/image\A_10_POM_IF_20221109_10_102155_01.jpg","./data/data/image\A_10_POM_IF_20221109_10_102155_02.jpg","./data/data/image\A_10_POM_IF_20221109_10_102155_03.jpg","./data/data/image\A_10_POM_IF_20221109_10_102155_08.jpg"]]
    start_time = time.time()
    probability = inference(model_path, image_paths)
    for i in range(len(probability)):
        if i == 0 :
            weight = "under weight"
        elif i == 1 :
            weight = "normal weight"
        else :
            weight = "over weight"
        
        print(f"{weight} probability : {100 * probability[i] : .4f} % ")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"elapsed time: {elapsed_time:.4f} sec")