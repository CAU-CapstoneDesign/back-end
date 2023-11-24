from django.urls import path
from .views import CreateObesityHistory, ObesityHistoryList

urlpatterns = [
    path('create_obesity_history/<int:pet_id>/', CreateObesityHistory.as_view(), name='create_obesity_history'),
    path('obesity_history/<int:pet_id>/', ObesityHistoryList.as_view(), name='obesity_history'),
]