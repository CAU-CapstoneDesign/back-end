# Generated by Django 4.2.7 on 2023-11-03 09:18

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ('pet', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='Disease',
            fields=[
                ('created_at', models.DateTimeField(auto_now_add=True, verbose_name='생성 일시')),
                ('updated_at', models.DateTimeField(auto_now=True, verbose_name='수정 일시')),
                ('name', models.CharField(max_length=20, primary_key=True, serialize=False, verbose_name='질병')),
                ('description', models.TextField(verbose_name='질병 설명')),
                ('treatment', models.TextField(verbose_name='일반적인 치료법')),
            ],
            options={
                'abstract': False,
            },
        ),
        migrations.CreateModel(
            name='DiseaseHistory',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('created_at', models.DateTimeField(auto_now_add=True, verbose_name='생성 일시')),
                ('updated_at', models.DateTimeField(auto_now=True, verbose_name='수정 일시')),
                ('diagnosis_date', models.DateField(verbose_name='진단 일자')),
                ('disease', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='disease.disease', verbose_name='질병')),
                ('pet', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='pet.pet', verbose_name='반려 동물')),
            ],
            options={
                'abstract': False,
            },
        ),
    ]
