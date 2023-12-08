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

"""

Inference Method 

you may use this method in your code.

input
model_path : path of model weight format ( .pt or .pth )
image_paths : 2d list of image paths ( [[ "A_04.jpg ","A_05.jpg ","A_13.jpg ","A_10.jpg ","A_09.jpg "  ] ,[ "B_04.jpg ","B_05.jpg ","B_13.jpg ","B_10.jpg ","B_09.jpg "  ]] )

output 
predictions = [[ A probability of low , A probability of normal , A probability of over   ] , [ B probability of low ,  B probability of normal ,B  probability of over   ] ...] 



"""


def inference(model_path, image_paths,age):

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
    
    with torch.no_grad():
        for image_path in image_paths:
            imagestack = []
            print(image_path)
            for angle_image_path in image_path :

                print(angle_image_path)

                image = Image.open(angle_image_path)
                image = transform(image).unsqueeze(0).to(device)
                imagestack.append(image)
            imagestack = torch.stack(imagestack, dim=2)
            age = torch.tensor(age, dtype=torch.float32).to(device).unsqueeze(0)

            logits = model(imagestack,age)
            probabilities = F.softmax(logits, dim=1)
            predictions.extend(probabilities.tolist())
    return predictions[0]

"""

Main Method

you can try inference method.
And output it as a percentage ( % ) 

"""

if __name__ == "__main__": 
    model_path = "./checkpoints/data1205POMBody4513109Age_mobilenetv21xkinetics_BCS3_19.pt"
    object_id = "over2"
    where = "inference"
    age =9


    image_paths = [[f"./data/data/{where}/{object_id}_04.jpg",f"./data/data/{where}/{object_id}_05.jpg",f"./data/data/{where}/{object_id}_13.jpg",f"./data/data/{where}/{object_id}_10.jpg",f"./data/data/{where}/{object_id}_09.jpg"]]

    
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
