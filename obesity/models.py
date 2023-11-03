from django.db import models
from owner.models import BaseModel
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

class ObesityHistory(BaseModel):
    pet = models.ForeignKey(Pet, verbose_name='반려 동물', on_delete=models.CASCADE)
    obesity = models.ForeignKey(Obesity, verbose_name='비만', on_delete=models.CASCADE)
    diagnosis_date = models.DateField(verbose_name='진단 일자')
    explanation = models.TextField(verbose_name='보호자 기록', blank=True)