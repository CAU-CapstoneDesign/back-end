# Generated by Django 4.2.7 on 2023-11-05 08:00

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('disease', '0003_remove_diseasehistory_created_at_and_more'),
    ]

    operations = [
        migrations.AlterField(
            model_name='diseasehistory',
            name='diagnosis_date',
            field=models.DateTimeField(auto_now=True, verbose_name='진단 일자'),
        ),
    ]
