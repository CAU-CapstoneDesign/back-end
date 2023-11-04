from django.contrib import admin
from .models import Obesity, ObesityHistory

# Register your models here.

admin.site.register(Obesity)
admin.site.register(ObesityHistory)