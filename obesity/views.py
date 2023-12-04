from django.shortcuts import render
from django.http import JsonResponse
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import datetime

from .models import Obesity, ObesityHistory
from .serializers import ObesitySerializer, ObesityHistorySerializer
from .inference_1203 import *
from pet.models import Pet

from django.core.files.storage import default_storage
import uuid

# Create your views here.

class CreateObesityHistory(APIView):
    def post(self, request, pet_id):
        uploaded_images = request.FILES.getlist('image')
        selected_breed = request.data.get('breed')
        age = request.data.get('age')

        if not uploaded_images:
            return JsonResponse({'error': 'Image files are required.'}, status=status.HTTP_400_BAD_REQUEST)
        
        if not selected_breed:
            return JsonResponse({'error': 'Breed is required.'}, status=status.HTTP_400_BAD_REQUEST)

        image_paths = []
        for uploaded_image in uploaded_images:
            extension = uploaded_image.name.split('.')[-1]
            unique_filename = f"{uuid.uuid4()}.{extension}"
            image_path = default_storage.save(f'obesity/{unique_filename}', uploaded_image)
            image_url = default_storage.url(image_path)
            image_paths.append(image_url)

        print(image_paths)

        start_time = time.time()
        # model_path = 'obesity/data1105POMBody1238_resnet183D_BCS3_2.pt'
        model_path = 'obesity/data1128POMBody4513109Age_mobilenetv21xkinetics_BCS3.pt'
        predictions = inference(model_path, image_paths, age)
            
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

        try:
            pet = Pet.objects.get(pk=pet_id)
        except Pet.DoesNotExist:
            return Response({'error': 'Pet not found'}, status=status.HTTP_404_NOT_FOUND)
        
        obesity_history = ObesityHistory(
            pet = pet,
            breed = selected_breed,
            age = age,
            result = result,
            diagnosis_date = datetime.date.today(),
            obesity_images=image_paths
        )
        obesity_history.save()

        serializer = ObesityHistorySerializer(obesity_history)
        return JsonResponse(serializer.data, status=status.HTTP_201_CREATED)
    
    '''
    def save_image(self, uploaded_images):
        # 이미지 저장 후 저장 경로 반환
        image_paths = []
        for uploaded_image in uploaded_images:
            image_path = 'obesity/input_images/' + uploaded_image.name
            
            with open(image_path, 'wb') as destination:
                for chunk in uploaded_image.chunks():
                    destination.write(chunk)
            image_paths.append(image_path)
        
        return image_paths    
    '''

class ObesityHistoryList(APIView):
    def get(self, request, pet_id):
        try:
            pet = Pet.objects.get(pk=pet_id)
        except Pet.DoesNotExist:
            return Response({'error': 'Pet not found'}, status=status.HTTP_404_NOT_FOUND)

        obesity_histories = ObesityHistory.objects.filter(pet=pet)
        serializer = ObesityHistorySerializer(obesity_histories, many=True)
        return Response(serializer.data)