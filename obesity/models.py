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
    pet_id = models.ForeignKey(Pet, verbose_name='반려 동물', on_delete=models.CASCADE)
    explanation = models.TextField(verbose_name='설명', blank=True)