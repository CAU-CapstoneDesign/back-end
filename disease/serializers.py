from rest_framework import serializers
from .models import Disease, DiseaseHistory

class DiseaseSerializer(serializers.ModelSerializer):
    class Meta:
        model = Disease
        fields = '__all__'

class DiseaseHistorySerializer(serializers.ModelSerializer):
    class Meta:
        model = DiseaseHistory
        fields = '__all__'