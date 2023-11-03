from django.contrib import admin
from .models import Disease, DiseaseHistory

# Register your models here.

admin.site.register(Disease)
admin.site.register(DiseaseHistory)