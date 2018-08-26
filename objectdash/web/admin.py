import uuid

from django.contrib import admin
from django.db import models
from django.forms import widgets

from objectdash.web.models import PBObjectDetector


@admin.register(PBObjectDetector)
class PBObjectDetectorAdmin(admin.ModelAdmin):
    fields = ('name', 'pb_file', 'label_file', 'active')
    list_display = ('name', 'active')
    formfield_overrides = {
        models.TextField: {'widget': widgets.TextInput},
    }

    def save_model(self, request, obj, form, change):
        if not obj.id:
            obj.id = uuid.uuid4()
        super().save_model(request, obj, form, change)
