from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('chat/', views.chat, name='chat'),
    path('api/ask/', views.ask, name='ask'),
    path('api/build-index/', views.build_index, name='build_index'),
]