from django.db import models
from accounts.models import BaseModel
from pet.models import Pet

# Create your models here.

class Obesity(BaseModel):
    LEVELS = (
        (1, '야윈 상태'),
        (3, '저체중'),
        (5, '이상적인 체중'),
        (7, '과체중'),
        (9, '비만')
    )

    level = models.IntegerField(choices=LEVELS)
    explanation = models.TextField(verbose_name='비만도 설명')

class ObesityHistory(models.Model):
    BREED = (
        ('Maltese', '말티즈'),
        ('Poodle', '푸들'),
        ('Bichon frise', '비숑 프리제'),
        ('Pomeranian', '포메라니안'),
        ('ETC', '기타')
    )

    pet = models.ForeignKey(Pet, verbose_name='반려 동물', on_delete=models.CASCADE)
    breed = models.CharField(choices=BREED, verbose_name='견종', max_length=15, null=False)
    # obesity = models.ForeignKey(Obesity, verbose_name='비만', on_delete=models.CASCADE)
    result = models.JSONField(verbose_name='예측 결과', default=dict)
    diagnosis_date = models.DateTimeField(verbose_name='진단 일자', auto_now=True)
    # explanation = models.TextField(verbose_name='보호자 기록', blank=True)