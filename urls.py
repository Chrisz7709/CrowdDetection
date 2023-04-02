from django.urls import path
from models import views
from .views import home


urlpatterns = [
    path('', home, name='home'),
]
