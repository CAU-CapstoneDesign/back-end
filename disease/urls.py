from django.urls import path
from .views import CreateDiseaseHistory, PetDiseaseHistoryList

urlpatterns = [
    path('create_disease_history/<int:pet_id>/', CreateDiseaseHistory.as_view(), name='create_disease_history'),
    path('disease_history/<int:pet_id>/', PetDiseaseHistoryList.as_view(), name='pet-disease-history'),
]