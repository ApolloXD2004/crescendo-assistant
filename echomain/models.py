from django.db import models
from .storage import OverwriteStorage
# Create your models here.



class Image(models.Model):
    image = models.ImageField(upload_to='',storage=OverwriteStorage())

   
