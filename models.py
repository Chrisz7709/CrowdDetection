from django.db import models

class CrowdData(models.Model):
    people_count = models.IntegerField()
    date_taken = models.DateTimeField()
    created_at = models.DateTimeField(auto_now_add=True)
