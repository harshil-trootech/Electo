from django.contrib import admin
from django.urls import path, include
from rest_framework import routers
from bill_prediction import views


app_name = 'bill_prediction'

router = routers.DefaultRouter()
router.register('api/bills', views.BillViewset)

urlpatterns = [
    path('', include(router.urls)),
]