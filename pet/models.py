from django.db import models
from owner.models import BaseModel, Owner

# Create your models here.

class Pet(BaseModel):
    BREED = (
        ('Maltese', '말티즈'),
        ('Poodle', '푸들'),
        ('Bichon frise', '비숑 프리제'),
        ('Pomeranian', '포메라니안'),
        ('ETC', '기타')
    )

    GENDER = (
        ('Female', '암컷'),
        ('Male', '수컷'),
        ('Neutered/Spayed', '중성화')
    )

    id = models.AutoField(primary_key=True)
    owner_id = models.ForeignKey(Owner, verbose_name='보호자', on_delete=models.CASCADE)
    name = models.CharField(verbose_name='이름', max_length=20, null=False)
    breed = models.CharField(choices=BREED, verbose_name='견종', max_length=20)
    gender = models.CharField(choices=GENDER, verbose_name='성별', max_length=20)
    # photo = models.ImageField(upload_to='', blank=True, null=True)        

    def __str__(self):
        return f'{self.owner_id}의 {self.name}'