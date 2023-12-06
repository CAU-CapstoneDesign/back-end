# Generated by Django 4.2.7 on 2023-12-06 11:16

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ('pet', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='Obesity',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('created_at', models.DateTimeField(auto_now_add=True, verbose_name='생성 일시')),
                ('updated_at', models.DateTimeField(auto_now=True, verbose_name='수정 일시')),
                ('level', models.IntegerField(choices=[(1, '야윈 상태'), (3, '저체중'), (5, '이상적인 체중'), (7, '과체중'), (9, '비만')])),
                ('explanation', models.TextField(verbose_name='비만도 설명')),
            ],
            options={
                'abstract': False,
            },
        ),
        migrations.CreateModel(
            name='ObesityHistory',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('breed', models.CharField(choices=[('Maltese', '말티즈'), ('Poodle', '푸들'), ('Bichon frise', '비숑 프리제'), ('Pomeranian', '포메라니안'), ('ETC', '기타')], max_length=15, verbose_name='견종')),
                ('age', models.IntegerField(verbose_name='나이')),
                ('result', models.JSONField(null=True, verbose_name='예측 결과')),
                ('diagnosis_date', models.DateTimeField(auto_now=True, verbose_name='진단 일자')),
                ('obesity_images', models.JSONField(blank=True, null=True, verbose_name='비만도 사진')),
                ('pet', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='pet.pet', verbose_name='반려 동물')),
            ],
        ),
    ]
