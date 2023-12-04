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
from .model_1203 import *
import boto3
import io
from PIL import Image
from config.settings import get_secret
from botocore.exceptions import ClientError

"""

Inference Method 

you may use this method in your code.

input
model_path : path of model weight format ( .pt or .pth )
image_paths : 2d list of image paths ( [[ "A_04.jpg ","A_05.jpg ","A_13.jpg ","A_10.jpg ","A_09.jpg "  ] ,[ "B_04.jpg ","B_05.jpg ","B_13.jpg ","B_10.jpg ","B_09.jpg "  ]] )

output 
predictions = [[ A probability of low , A probability of normal , A probability of over   ] , [ B probability of low ,  B probability of normal ,B  probability of over   ] ...] 



"""


def inference(model_path, image_paths, age):

    transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = get_model(num_classes=3, sample_size=64, width_mult=1.).to(device)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(model_path, map_location=device))
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
    bucket_name = 'petcare-capstone' 

    with torch.no_grad():
        imagestack = []

        for image_path in image_paths:
            bucket_name = 'petcare-capstone'
            key = '/'.join(image_path.split('/')[-2:])  # S3 객체 키 추출
            print(key)
            response = s3.get_object(Bucket=bucket_name, Key=key)
            image_bytes = response['Body'].read() 
            image = Image.open(io.BytesIO(image_bytes))
            image = transform(image).unsqueeze(0).to(device)
            imagestack.append(image)
            
            imagestack_tor = torch.stack(imagestack, dim=2)
            age = float(age)
            age_tensor = torch.tensor([age], dtype=torch.float32).to(device).unsqueeze(0)

            logits = model(imagestack_tor, age_tensor)
            probabilities = F.softmax(logits, dim=1)
            predictions.extend(probabilities.tolist())

        return predictions[0]

"""

Main Method

you can try inference method.
And output it as a percentage ( % ) 

"""

if __name__ == "__main__": 
    model_path = "./checkpoints/data1128POMBody4513109Age_mobilenetv21xkinetics_BCS3.pt"
    object_id = "A_10_POM_IM_20221111_10_000853"



    image_paths = [[f"./data/data/image\{object_id}_04.jpg",f"./data/data/image\{object_id}_05.jpg",f"./data/data/image\{object_id}_13.jpg",f"./data/data/image\{object_id}_10.jpg",f"./data/data/image\{object_id}_09.jpg"]]
    age = 10
    
    start_time = time.time()
    probability = inference(model_path, image_paths, age)
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