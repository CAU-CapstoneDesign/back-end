from django.urls import path
from .views import CreateDiseaseHistory

urlpatterns = [
    path('create_disease_history/<int:pet_id>/', CreateDiseaseHistory.as_view(), name='create_disease_history'),
]