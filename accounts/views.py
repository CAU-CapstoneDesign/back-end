from django.shortcuts import render
from django.http import JsonResponse
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from .models import User
from .serializers import UserSerializer

# Create your views here.

class UserList(APIView):
    def get(self, request):
        owners = User.objects.all()
        serializer = UserSerializer(owners, many=True)
        return Response(serializer.data)

    def post(self, request):
        serializer = UserSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class UserDetail(APIView):
    def get(self, request, pk):
        try:
            owner = User.objects.get(pk=pk)
        except User.DoesNotExist:
            return Response({'error': 'Owner not found'}, status=status.HTTP_404_NOT_FOUND)

        serializer = UserSerializer(owner)
        return Response(serializer.data)

    def put(self, request, pk):
        try:
            owner = User.objects.get(pk=pk)
        except User.DoesNotExist:
            return Response({'error': 'Owner not found'}, status=status.HTTP_404_NOT_FOUND)

        serializer = UserSerializer(owner, data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def delete(self, request, pk):
        try:
            owner = User.objects.get(pk=pk)
        except User.DoesNotExist:
            return Response({'error': 'Owner not found'}, status=status.HTTP_404_NOT_FOUND)

        owner.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)
    
from django.shortcuts import redirect
from django.conf import settings
from django.http import JsonResponse, HttpResponse
from accounts.models import User
from dj_rest_auth.registration.views import SocialLoginView
from rest_framework import status
from allauth.socialaccount.models import SocialAccount
from allauth.socialaccount.providers.google import views as google_view
from allauth.socialaccount.providers.oauth2.client import OAuth2Client
from json.decoder import JSONDecodeError
import requests

state = getattr(settings, 'STATE')
BASE_URL = 'http://localhost:8000/'
GOOGLE_CALLBACK_URI = BASE_URL + 'accounts/google/callback/'

def google_login(request):
    # Code Request
    scope = "https://www.googleapis.com/auth/userinfo.email"
    client_id = getattr(settings, "SOCIAL_AUTH_GOOGLE_CLIENT_ID")
    return redirect(f"https://accounts.google.com/o/oauth2/v2/auth?client_id={client_id}&response_type=code&redirect_uri={GOOGLE_CALLBACK_URI}&scope={scope}")

def google_callback(request):
    client_id = getattr(settings, "SOCIAL_AUTH_GOOGLE_CLIENT_ID")
    client_secret = getattr(settings, "SOCIAL_AUTH_GOOGLE_SECRET")
    code = request.GET.get('code')

    # Access Token Request
    token_req = requests.post(
        f"https://oauth2.googleapis.com/token?client_id={client_id}&client_secret={client_secret}&code={code}&grant_type=authorization_code&redirect_uri={GOOGLE_CALLBACK_URI}&state={state}")
    token_req_json = token_req.json()
    error = token_req_json.get("error")
    if error is not None:
        raise JSONDecodeError(error)
    access_token = token_req_json.get('access_token')
    
    # Email Request
    email_req = requests.get(
        f"https://www.googleapis.com/oauth2/v1/tokeninfo?access_token={access_token}")
    email_req_status = email_req.status_code
    if email_req_status != 200:
        return JsonResponse({'err_msg': 'failed to get email'}, status=status.HTTP_400_BAD_REQUEST)
    email_req_json = email_req.json()
    email = email_req_json.get('email')

    response_data = {
        'email': email,
        'access_token': access_token,
        'code': code,
    }

    # Signup or Signin Request
    try:
        user = User.objects.get(email=email)
        # 다른 SNS로 가입된 유저
        social_user = SocialAccount.objects.get(user=user)
        if social_user is None:
            return JsonResponse({'err_msg': 'email exists but not social user'}, status=status.HTTP_400_BAD_REQUEST)
        if social_user.provider != 'github':
            return JsonResponse({'err_msg': 'no matching social type'}, status=status.HTTP_400_BAD_REQUEST)
        # 기존에 github로 가입된 유저
        data = {'access_token': access_token, 'code': code}
        accept = requests.post(
            f"{BASE_URL}accounts/github/login/finish/", data=data)
        accept_status = accept.status_code
        if accept_status != 200:
            return JsonResponse({'err_msg': 'failed to signin'}, status=accept_status)
        accept_json = accept.json()
        accept_json.pop('user', None)
        return JsonResponse(accept_json)
    except User.DoesNotExist:
        # 기존에 가입된 유저가 없으면 새로 가입
        data = {'access_token': access_token, 'code': code}
        accept = requests.post(
            f"{BASE_URL}accounts/github/login/finish/", data=data)
        accept_status = accept.status_code
        if accept_status != 200:
            return JsonResponse({'err_msg': 'failed to signup'}, status=accept_status)
        # user의 pk, email, first name, last name과 Access Token, Refresh token 가져옴
        accept_json = accept.json()
        accept_json.pop('user', None)
        return JsonResponse(accept_json)

class GoogleLogin(SocialLoginView):
    adapter_class = google_view.GoogleOAuth2Adapter
    callback_url = GOOGLE_CALLBACK_URI
    client_class = OAuth2Client