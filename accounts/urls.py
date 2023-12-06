from django.urls import path, include
from .views import *
from rest_framework import urls

urlpatterns = [
    path('', UserList.as_view(), name='owner_list'),
    path('<int:pk>/', UserDetail.as_view(), name='owner_detail'),
    # path('google/login/', google_login, name='google_login'),
    # path('google/callback/', google_callback, name='google_callback'),
    # path('google/login/finish/', GoogleLogin.as_view(), name='google_login_todjango'),    
    path('signup/', UserCreate.as_view()),
    path('login/', LoginView.as_view(), name='login'),
    path('logout/', LogoutView.as_view(), name='logout'),
    path('api-auth/', include('rest_framework.urls')),
]
