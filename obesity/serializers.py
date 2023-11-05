from rest_framework import serializers
from .models import Obesity, ObesityHistory

class ObesitySerializer(serializers.ModelSerializer):
    class Meta:
        model = Obesity
        fields = '__all__'

class ObesityHistorySerializer(serializers.ModelSerializer):
    class Meta:
        model = ObesityHistory
        fields = '__all__'