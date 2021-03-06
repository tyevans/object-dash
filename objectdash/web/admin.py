from django.contrib import admin
from django.db import models
from django.forms import widgets
from django.utils.safestring import mark_safe

from objectdash.web.models import ObjectDetectionModel, AnnotatedImage


@admin.register(ObjectDetectionModel)
class PBObjectDetectorAdmin(admin.ModelAdmin):
    fields = ('name', 'pb_file', 'label_file', 'num_classes', 'active')
    list_display = ('name', 'active')
    list_editable = ('active',)
    formfield_overrides = {
        models.TextField: {'widget': widgets.TextInput},
    }


@admin.register(AnnotatedImage)
class AnnotatedImageAdmin(admin.ModelAdmin):
    fields = ('image_file', )
    list_display = ('image_file_thumbnail', 'get_annotation_labels', 'source')
    list_filter = ('annotations__label__label', 'source')

    def image_file_thumbnail(self, instance):
        return mark_safe('<img width=100 src="{}">'.format(instance.image_file.url))

    image_file_thumbnail.short_description = "Thumbnail"

    def get_annotation_labels(self, instance):
        if instance.annotations:
            return ", ".join(anno.label.label for anno in instance.annotations.all())
        return ""

    get_annotation_labels.short_description = "Annotations"