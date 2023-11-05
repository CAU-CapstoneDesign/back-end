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
    
class DiseaseHistory(models.Model):
    PARTS = (
        ('Head', '머리'),
        ('Body', '몸통'),
        ('Leg', '다리'),
        ('Tail', '연접부')
    )

    # photo = models.ImageField(upload_to='', blank=True, null=True)
    pet = models.ForeignKey(Pet, verbose_name='반려 동물', on_delete=models.CASCADE)
    part = models.CharField(choices=PARTS, verbose_name='질환 의심 부위', max_length=10, null=False)
    result = models.JSONField(verbose_name='예측 결과', default=dict)
    # disease = models.ForeignKey(Disease, verbose_name='질병', on_delete=models.CASCADE)
    diagnosis_date = models.DateTimeField(verbose_name='진단 일자', auto_now=True)
    # explanation = models.TextField(verbose_name='보호자 기록', blank=True)