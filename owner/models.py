from django.db import models

# Create your models here.

class BaseModel(models.Model):
    created_at = models.DateTimeField(verbose_name='생성 일시', auto_now_add=True)
    updated_at = models.DateTimeField(verbose_name='수정 일시', auto_now=True)

    class Meta:
        abstract = True

class Owner(BaseModel):
    id = models.AutoField(primary_key=True)
    name = models.CharField(verbose_name='이름', max_length=15, null=False)
    nickname = models.CharField(verbose_name='별명', max_length=20, null=False, unique=True)
    phone_number = models.CharField(verbose_name='전화번호', max_length=11, null=True)
    address = models.CharField(verbose_name='주소', max_length=50, null=True)

    def __str__(self):
        return self.nickname