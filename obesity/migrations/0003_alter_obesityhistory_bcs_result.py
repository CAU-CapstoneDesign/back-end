# Generated by Django 4.2.7 on 2023-12-08 01:58

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('obesity', '0002_remove_obesityhistory_result_and_more'),
    ]

    operations = [
        migrations.AlterField(
            model_name='obesityhistory',
            name='bcs_result',
            field=models.FloatField(blank=True, null=True, verbose_name='BCS 예측 결과'),
        ),
    ]
