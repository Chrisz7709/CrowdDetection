from django.db import models

class CrowdData(models.Model):
    people_count = models.IntegerField()
    date_taken = models.DateTimeField()
    timestamp = models.DateTimeField(auto_now_add=True)  
    image = models.ImageField(upload_to='images/') 