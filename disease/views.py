from django.shortcuts import render
from django.http import JsonResponse
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import datetime

from .models import Disease, DiseaseHistory
from .serializers import DiseaseSerializer, DiseaseHistorySerializer
from .inference import *
from pet.models import Pet

# Create your views here.

class CreateDiseaseHistory(APIView):
    def post(self, request, pet_id):
        uploaded_image = request.data.get('image')
        selected_part = request.data.get('part')

        if uploaded_image and selected_part:
            image_path = self.save_image(uploaded_image)

            # 모델을 통한 예측 수행
            model_path = 'disease/data0922_resnet18_dogB_0.pt'
            predictions = inference(model_path, [image_path])

            try:
                pet = Pet.objects.get(pk=pet_id)
            except Pet.DoesNotExist:
                return Response({'error': 'Pet not found'}, status=status.HTTP_404_NOT_FOUND)

            # DiseaseHistory 모델에 새로운 엔트리 생성
            disease_history = DiseaseHistory(
                pet = pet,
                part = selected_part,
                result = predictions,
                diagnosis_date = datetime.date.today()
            )
            disease_history.save()

            serializer = DiseaseHistorySerializer(disease_history)

            return JsonResponse(serializer.data, status=status.HTTP_201_CREATED)
        
        return JsonResponse({'error': 'POST request with an image and selected_part required'}, status=status.HTTP_400_BAD_REQUEST)

    def save_image(self, uploaded_image):
        # 이미지 저장 후 저장 경로 반환
        image_path = 'disease/input_images/' + uploaded_image.name
        with open(image_path, 'wb') as destination:
            for chunk in uploaded_image.chunks():
                destination.write(chunk)
        return image_path