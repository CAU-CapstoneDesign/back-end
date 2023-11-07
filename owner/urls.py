from django.urls import path
from .views import OwnerList, OwnerDetail

urlpatterns = [
    path('', OwnerList.as_view(), name='owner_list'),
    path('<int:pk>/', OwnerDetail.as_view(), name='owner_detail'),
]
