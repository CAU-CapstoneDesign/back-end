# Generated by Django 4.2.7 on 2023-12-08 01:56

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('obesity', '0001_initial'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='obesityhistory',
            name='result',
        ),
        migrations.AddField(
            model_name='obesityhistory',
            name='bcs_result',
            field=models.JSONField(blank=True, null=True, verbose_name='BCS 예측 결과'),
        ),
        migrations.AddField(
            model_name='obesityhistory',
            name='obesity_result',
            field=models.JSONField(null=True, verbose_name='비만도 예측 결과'),
        ),
    ]