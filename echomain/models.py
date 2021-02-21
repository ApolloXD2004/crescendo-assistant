from django.db import models
from .storage import OverwriteStorage
# Create your models here.



def get_path_and_name(instance, filename):
     new_name = 'blabla.jpg' 
     return new_name

class Image(models.Model):
    image = models.ImageField(upload_to=get_path_and_name,storage=OverwriteStorage())

   
