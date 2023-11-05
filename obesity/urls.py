from django.urls import path
from .views import CreateObesityHistory

urlpatterns = [
    path('create_obesity_history/<int:pet_id>/', CreateObesityHistory.as_view(), name='create_obesity_history'),
]