from django.db import models
from owner.models import BaseModel, Owner

# Create your models here.

class Pet(BaseModel):
    BREED = (
        ('Retriever', '리트리버'),
        ('Pomeranian', '포메라니안'),
        ('Poodle', '푸들')
    )

    GENDER = (
        ('Female', '암컷'),
        ('Male', '수컷'),
        ('Neutered/Spayed', '중성화')
    )

    id = models.AutoField(primary_key=True)
    owner_id = models.ForeignKey(Owner, verbose_name='보호자', on_delete=models.CASCADE)
    breed = models.CharField(choices=BREED, max_length=20)
    gender = models.CharField(choices=GENDER, max_length=20)
    # photo = models.ImageField(upload_to='', blank=True, null=True)