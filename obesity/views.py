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

# Create your views here.

class CreateObesityHistory(APIView):
    def post(self, request, pet_id):
        uploaded_images = request.FILES.getlist('image')
        selected_breed = request.data.get('breed')

        if uploaded_images and selected_breed:
            image_paths = self.save_image(uploaded_images)

            model_path = 'obesity/data1105POMBody1238_resnet183D_BCS3_2.pt'
            predictions = inference(model_path, [image_paths])
            
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

            # DiseaseHistory 모델에 새로운 엔트리 생성
            obesity_history = ObesityHistory(
                pet = pet,
                breed = selected_breed,
                result = result,
                diagnosis_date = datetime.date.today()
            )
            obesity_history.save()

            serializer = ObesityHistorySerializer(obesity_history)

            return JsonResponse(serializer.data, status=status.HTTP_201_CREATED)
        
        return JsonResponse({'error': 'POST request with an image and selected_breed required'}, status=status.HTTP_400_BAD_REQUEST)        

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