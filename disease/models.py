from django.db import models
from owner.models import BaseModel
from pet.models import Pet

# Create your models here.

class Disease(BaseModel):
    name = models.CharField(primary_key=True, verbose_name='질병', max_length=20)
    description = models.TextField(verbose_name='질병 설명')
    treatment = models.TextField(verbose_name='일반적인 치료법')

    def __str__(self):
        return self.name
    
class DiseaseHistory(BaseModel):
    PARTS = (
        ('Head', '머리'),
        ('Body', '몸통'),
        ('Leg', '다리'),
        ('', '연접부')
    )

    pet = models.ForeignKey(Pet, verbose_name='반려 동물', on_delete=models.CASCADE)
    part = models.CharField(choices=PARTS, verbose_name='질환 의심 부위', max_length=10, null=False)
    disease = models.ForeignKey(Disease, verbose_name='질병', on_delete=models.CASCADE)
    diagnosis_date = models.DateField(verbose_name='진단 일자')
    explanation = models.TextField(verbose_name='보호자 기록', blank=True)