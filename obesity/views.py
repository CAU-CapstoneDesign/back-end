from django.shortcuts import render
from django.http import JsonResponse
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import datetime

from .models import Obesity, ObesityHistory
from .serializers import ObesitySerializer, ObesityHistorySerializer
from .inference_1102 import *
from pet.models import Pet

# Create your views here.

class CreateObesityHistory(APIView):
    def post(self, request, pet_id):
        uploaded_image = request.data.get('image')
        selected_breed = request.data.get('breed')

        if uploaded_image and selected_breed:
            image_path = self.save_image(uploaded_image)

            model_path = 'obesity/data1102POMAll_resnet18_BCS3_00.pt'
            predictions = inference(model_path, [image_path])

            try:
                pet = Pet.objects.get(pk=pet_id)
            except Pet.DoesNotExist:
                return Response({'error': 'Pet not found'}, status=status.HTTP_404_NOT_FOUND)

            # DiseaseHistory 모델에 새로운 엔트리 생성
            obesity_history = ObesityHistory(
                pet = pet,
                breed = selected_breed,
                result = predictions,
                diagnosis_date = datetime.date.today()
            )
            obesity_history.save()

            serializer = ObesityHistorySerializer(obesity_history)

            return JsonResponse(serializer.data, status=status.HTTP_201_CREATED)
        
        return JsonResponse({'error': 'POST request with an image and selected_breed required'}, status=status.HTTP_400_BAD_REQUEST)        

    def save_image(self, uploaded_image):
        # 이미지 저장 후 저장 경로 반환
        image_path = 'obesity/input_images/' + uploaded_image.name
        with open(image_path, 'wb') as destination:
            for chunk in uploaded_image.chunks():
                destination.write(chunk)
        return image_path    