import random

import tensorflow as tf
from django.core.paginator import Paginator
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


class TFRecordExportView(TemplateView):
    template_name = "object_detection/annotation_browser.html"

    def get_records(self, form_data):
        images = AnnotatedImage.objects.prefetch_related('annotations', 'annotations__label')

        labels = form_data.get('labels')
        if labels:
            if not isinstance(labels, list):
                labels = [labels]
        else:
            labels = []
        if labels:
            images = images.filter(annotations__label__in=labels)

        source = form_data.get('source')
        if source:
            images = images.filter(source=source)

        return images

    def write_tfrecord(self, images, record_path):
        writer = tf.python_io.TFRecordWriter(record_path)
        for image in images:
            writer.write(image.tf_example().SerializeToString())
        writer.close()

    def render_label_pbtxt(self, label_map):
        template = """\
        item {{
          name: "{0}"
          id: {1}
        }}"""

        output = "\n".join(template.format(*item) for item in label_map.items())
        return output

    def get(self, request, *args, **kwargs):
        form = AnnotationFilterForm(request.GET)
        if form.is_valid():
            images = self.get_records(form.cleaned_data)
            shuffled_images = list(images.all())
            random.shuffle(shuffled_images)

            label_map = {}

            images_seen = set()  # lazy dedupe
            num_images = len(shuffled_images)
            unique_images = []
            for image in shuffled_images:
                if image.id in images_seen:
                    continue
                images_seen.add(image.id)
                unique_images.append(image)

                for annotation in image.annotations.all():
                    if str(annotation.label.id) in form.cleaned_data['labels']:
                        label_map[annotation.label.label] = annotation.label.id

            num_train = int(num_images * 0.7)
            train_images, eval_images = shuffled_images[:num_train], shuffled_images[num_train:]

            with open("labels.pbtxt", 'w') as fd:
                fd.write(self.render_label_pbtxt(label_map))
            self.write_tfrecord(train_images, 'train.tfrecord')
            self.write_tfrecord(eval_images, 'eval.tfrecord')

        else:
            images = AnnotatedImage.objects.all()

        page = request.GET.get('page', 1)
        paginator = Paginator(images, 25)
        images_page = paginator.get_page(page)

        return self.render_to_response({
            "form": form,
            "images_page": images_page
        })
