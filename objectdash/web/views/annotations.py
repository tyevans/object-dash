import tensorflow as tf
from django.core.paginator import Paginator
from django.views import View
from django.views.generic import TemplateView

from objectdash.web.forms import AnnotationFilterForm
from objectdash.web.models import AnnotatedImage


class AnnotationListView(TemplateView):
    template_name = "object_detection/annotation_browser.html"

    def form_valid(self, request, form):
        cleaned = form.cleaned_data

        images = AnnotatedImage.objects

        labels = cleaned.get('labels')
        if labels:
            if not isinstance(labels, list):
                labels = [labels]
        else:
            labels = []
        if labels:
            images = AnnotatedImage.objects.filter(annotations__label__in=labels)

        source = cleaned.get('source')
        if source:
            images = images.filter(source=source)
        return images

    def handle_form(self, request, form):
        if form.is_valid():
            images = self.form_valid(request, form)
        else:
            images = AnnotatedImage.objects
        page = request.GET.get('page', 1)
        paginator = Paginator(images.all(), 25)
        images_page = paginator.get_page(page)

        return self.render_to_response({
            "form": form,
            "images_page": images_page
        })

    def get(self, request, *args, **kwargs):
        form = AnnotationFilterForm(request.GET)
        return self.handle_form(request, form)

    def post(self, request, *args, **kwargs):
        form = AnnotationFilterForm(request.POST, request.FILES)
        return self.handle_form(request, form)


class TFRecordExportView(View):
    template_name = "object_detection/annotation_browser.html"

    def get_records(self, form_data):
        images = AnnotatedImage.objects

        labels = form_data.get('labels')
        if labels:
            if not isinstance(labels, list):
                labels = [labels]
        else:
            labels = []
        if labels:
            images = AnnotatedImage.objects.filter(annotations__label__in=labels)

        source = form_data.get('source')
        if source:
            images = images.filter(source=source)
        return images

    def get(self, request, *args, **kwargs):
        form = AnnotationFilterForm(request.GET)
        if form.is_valid():
            images = self.get_records(form.cleaned_data)

            writer = tf.python_io.TFRecordWriter("output.data")
            for image in images.all():
                writer.write(image.tf_example().SerializeToString())
            writer.close()
