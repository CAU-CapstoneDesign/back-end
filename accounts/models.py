from django.contrib.auth.models import AbstractUser
from django.db import models

class User(AbstractUser):
    nickname = models.CharField(verbose_name='별명', max_length=20, null=False, unique=True)
    phone_number = models.CharField(verbose_name='전화번호', max_length=11, null=True)
    address = models.CharField(verbose_name='주소', max_length=50, null=True)

    def __str__(self):
        return self.nickname
    
class BaseModel(models.Model):
    created_at = models.DateTimeField(verbose_name='생성 일시', auto_now_add=True)
    updated_at = models.DateTimeField(verbose_name='수정 일시', auto_now=True)

    class Meta:
        abstract = True    