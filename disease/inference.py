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


"""

Model Define

input : batch x 3 x 64 x 64
output : num_classes ( 7 )


"""

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResNet18(nn.Module):
    def __init__(self, num_classes=1):
        super(ResNet18, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        layers = []
        strides = [stride] + [1] * (num_blocks - 1)
        for stride in strides:
            layers.append(ResidualBlock(in_channels, out_channels, stride))
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
image_paths : 1d list of image paths ( [ "./~~/~~/~~.jpg " ,"./~~/~~/~~.jpg " ,"./~~/~~/~~.jpg " ] )

output 
probabilities 2d list [  [probability of 1th picture]   [probability of 2th picture] [probability of 3th picture]   ] 
[probability of 1th picture] = [  probability of 1th disease ,  probability of 2th disease ...   probability of 7th disease   ]



"""

import boto3
import io
from config.settings import get_secret


def inference(model_path, image_paths):

    transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ResNet18(num_classes=7).to(device)

    # model.load_state_dict(torch.load(model_path))
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    
    predictions = []

    aws_access_key_id = get_secret('AWS_ACCESS_KEY_ID')
    aws_secret_access_key = get_secret('AWS_SECRET_ACCESS_KEY')
    aws_region = get_secret('AWS_REGION')
    
    s3 = boto3.client(
        's3',
        region_name=aws_region,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key
    )

    with torch.no_grad():
        for image_path in image_paths:
            # S3에서 이미지 다운로드
            bucket_name = 'cau-petdoctor'
            response = s3.get_object(Bucket=bucket_name, Key=image_path)
            image_bytes = response['Body'].read()
            image = Image.open(io.BytesIO(image_bytes))
            
            image = transform(image).unsqueeze(0).to(device)
            logits = model(image)
            probabilities = F.softmax(logits, dim=1)
            predictions.extend(probabilities.tolist())

    return predictions



"""

Main Method

you can try inference method.
In this example, image_paths is assigned from the csv file. 
And output it as a percentage ( % ) 

"""

if __name__ == "__main__": 
    model_path = "./checkpoints/data0922_resnet18_dogB_0.pt"
    test_path = "./data/data/dog/test_1.csv"
    test_df = pd.read_csv(test_path)
    image_paths = test_df["path"].tolist()

    probabilities = inference(model_path, image_paths)
    for probability in probabilities:
        print([100 * p for p in probability])
