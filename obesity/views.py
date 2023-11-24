from django.shortcuts import render
from django.http import JsonResponse
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import datetime

from .models import Obesity, ObesityHistory
from .serializers import ObesitySerializer, ObesityHistorySerializer
from .inference_1108 import *
from pet.models import Pet

from django.core.files.storage import default_storage

# Create your views here.

class CreateObesityHistory(APIView):
    def post(self, request, pet_id):
        uploaded_images = request.FILES.getlist('image')
        selected_breed = request.data.get('breed')

        if not uploaded_images:
            return JsonResponse({'error': 'Image files are required.'}, status=status.HTTP_400_BAD_REQUEST)
        
        if not selected_breed:
            return JsonResponse({'error': 'Breed is required.'}, status=status.HTTP_400_BAD_REQUEST)

        image_paths = [default_storage.save('obesity/' + uploaded_image.name, uploaded_image)
                       for uploaded_image in uploaded_images]

        print(image_paths)

        model_path = 'obesity/data1105POMBody1238_resnet183D_BCS3_2.pt'
        predictions = inference(model_path, image_paths)
            
        result = [max(predictions)]
        index = predictions.index(result[0])
        if index == 0:
            result.append('under weight')
        elif index == 1:
            result.append('normal weight')
        elif index == 2:
            result.append('over weight')

        try:
            pet = Pet.objects.get(pk=pet_id)
        except Pet.DoesNotExist:
            return Response({'error': 'Pet not found'}, status=status.HTTP_404_NOT_FOUND)

        image_paths = [default_storage.url(default_storage.save('obesity/' + uploaded_image.name, uploaded_image))
              for uploaded_image in uploaded_images]
        
        obesity_history = ObesityHistory(
            pet = pet,
            breed = selected_breed,
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