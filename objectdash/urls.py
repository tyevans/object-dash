"""objectdash URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import path

import objectdash.web.views.annotations as anno_views
import objectdash.web.views.object_detection as od_views
from objectdash.web.views import index as index_views

urlpatterns = [
    path('admin/', admin.site.urls),
    path("", index_views.IndexView.as_view(), name="index"),
    path("object_detection/single_image",
         od_views.SingleImageObjectDetectionView.as_view(), name="single-image-object-detection"),
    path("object_detection/annotation_browser",
         anno_views.AnnotationListView.as_view(), name="annotation-browser"),
    path("object_detection/tf_record_export",
         anno_views.TFRecordExportView.as_view(), name="tf-record-export"),
]

if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
