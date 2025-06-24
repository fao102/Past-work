from django.urls import path
from spotify_control import views

urlpatterns = [
    path("", views.main, name="main"),
]
