# Generated by Django 4.2.7 on 2023-11-05 07:57

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('obesity', '0002_remove_obesityhistory_explanation_and_more'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='obesityhistory',
            name='created_at',
        ),
        migrations.RemoveField(
            model_name='obesityhistory',
            name='updated_at',
        ),
        migrations.AlterField(
            model_name='obesityhistory',
            name='diagnosis_date',
            field=models.DateTimeField(verbose_name='진단 일자'),
        ),
    ]
