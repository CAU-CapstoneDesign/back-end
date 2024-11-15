from django.shortcuts import render
from django.http import JsonResponse
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import datetime

from .models import Obesity, ObesityHistory
from .serializers import ObesitySerializer, ObesityHistorySerializer
from .inference_1203 import *
from .inference_1207 import *
from pet.models import Pet
import os
import boto3
from PIL import Image
from config.settings import get_secret
from django.conf import settings
from botocore.exceptions import NoCredentialsError
import uuid

class CreateObesityHistory(APIView):
    def post(self, request, pet_id):
        uploaded_images = request.FILES.getlist('image')
        selected_breed = request.data.get('breed')
        age = request.data.get('age')

        try:
            pet = Pet.objects.get(pk=pet_id)
        except Pet.DoesNotExist:
            return Response({'error': 'Pet not found'}, status=status.HTTP_404_NOT_FOUND)

        if not uploaded_images:
            return JsonResponse({'error': 'Image files are required.'}, status=status.HTTP_400_BAD_REQUEST)
        
        if not selected_breed:
            return JsonResponse({'error': 'Breed is required.'}, status=status.HTTP_400_BAD_REQUEST)

        image_paths = []
        for uploaded_image in uploaded_images:
            image_path = self.save_image(uploaded_image)
            image_paths.append(image_path)

        print([image_paths])

        start_time = time.time()

        model_path_obesity = 'obesity/data1205POMBody4513109Age_mobilenetv21xkinetics_BCS3_19.pt'
        predictions = inference_obesity(model_path_obesity, [image_paths], int(age))

        model_path_bcs = 'obesity/data1205POMBody4513109Age_mobilenetv21xkinetics_BCS_weightedMSELoss_19.pt'
        bcs = inference_bcs(model_path_bcs, [image_paths], int(age))

        print(predictions)

        result = [max(predictions)]
        index = predictions.index(result[0])
        if index == 0:
            result.append('under weight')
        elif index == 1:
            result.append('normal weight')
        elif index == 2:
            result.append('over weight')

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"elapsed time: {elapsed_time:.4f} sec")
        
        obesity_history = ObesityHistory(
            pet = pet,
            breed = selected_breed,
            age = age,
            obesity_result = result,
            bcs_result = bcs,
            diagnosis_date = datetime.date.today(),
            obesity_images=[]
        )
        obesity_history.save()

        # AWS S3
        aws_access_key_id = get_secret('AWS_ACCESS_KEY_ID')
        aws_secret_access_key = get_secret('AWS_SECRET_ACCESS_KEY')
        aws_region = get_secret('AWS_REGION')
        
        s3 = boto3.client(
            's3',
            region_name=aws_region,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key
        )
        bucket_name = 'cau-petdoctor' 

        s3_image_paths = []
        for temp_image_path in image_paths:
            unique_id = uuid.uuid4()
            s3_file_key = f"obesity/{unique_id}.{os.path.basename(temp_image_path)}"

            try:
                with open(temp_image_path, 'rb') as data:
                    s3.upload_fileobj(data, bucket_name, s3_file_key)
                s3_image_url = f"https://{bucket_name}.s3.{aws_region}.amazonaws.com/{s3_file_key}"
                s3_image_paths.append(s3_image_url)
                # 업로드 후 로컬 파일 삭제
                os.remove(temp_image_path)
            except NoCredentialsError:
                print("Credentials not available")
                continue
        
        print(s3_image_paths)

        obesity_history.obesity_images = s3_image_paths
        obesity_history.save()

        serializer = ObesityHistorySerializer(obesity_history)
        return JsonResponse(serializer.data, status=status.HTTP_201_CREATED)
    
    def save_image(self, uploaded_image):
        # 이미지 저장 후 상대 경로 반환
        image_name = uploaded_image.name
        image_path = os.path.join('input_images', image_name)
        with open(image_path, 'wb') as destination:
            for chunk in uploaded_image.chunks():
                destination.write(chunk)
        return image_path

class ObesityHistoryList(APIView):
    def get(self, request, pet_id):
        try:
            pet = Pet.objects.get(pk=pet_id)
        except Pet.DoesNotExist:
            return Response({'error': 'Pet not found'}, status=status.HTTP_404_NOT_FOUND)

        obesity_histories = ObesityHistory.objects.filter(pet=pet)
        serializer = ObesityHistorySerializer(obesity_histories, many=True)
        return Response(serializer.data)