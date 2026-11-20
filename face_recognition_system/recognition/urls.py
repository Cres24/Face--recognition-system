from django.urls import path

from . import views

app_name = "recognition"

urlpatterns = [
    path("faces/", views.register_face, name="register-face"),
    path("recognize/", views.recognize_face, name="recognize-face"),
]