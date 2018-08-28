import os
import random
import time
import uuid

import cv2
import numpy as np
from django.conf import settings
from django.contrib.staticfiles.templatetags.staticfiles import static
from django.core.paginator import Paginator
from django.http import JsonResponse
from django.utils.text import slugify
from django.views import View
from django.views.generic import TemplateView

from objectdash.web.forms import ClassifyImageForm, VerificationReportForm
from objectdash.web.models import PBObjectDetector, ExampleImage


def classify_image_from_bytes(image_np):
    detectors = PBObjectDetector.objects.filter(active=True).all()
    annotations = {}
    for detector in detectors:
        start_time = time.time()
        annotations[detector.name] = {
            "results": detector.annotate(image_np)
        }
        annotations[detector.name]['runtime'] = time.time() - start_time
    return annotations


def visualize_annotation(image_np, annotations):
    for annotation in annotations:
        annotation.draw(image_np, get_random_color())
    image_id = str(uuid.uuid4())
    image_path = os.path.join("vis/annotations/", image_id + '.jpg')
    image_name = os.path.join(settings.STATIC_ROOT, image_path)
    cv2.imwrite(image_name, image_np)
    return static(image_path)


def crop_annotations(image_np, annotations):
    crops = []
    for annotation in annotations:
        crop_image = annotation.crop(image_np)
        image_id = str(uuid.uuid4())
        image_path = os.path.join("vis/annotations/crops/", image_id + '.jpg')
        image_name = os.path.join(settings.STATIC_ROOT, image_path)
        cv2.imwrite(image_name, crop_image)
        crops.append(static(image_path))
    return crops


def get_random_color(num_channels=3):
    x = list(range(0, 255))
    if num_channels == 1:
        return [random.choice(x)]
    if num_channels == 3:
        return [random.choice(x), random.choice(x), random.choice(x)]
    if num_channels == 4:
        return [random.choice(x), random.choice(x), random.choice(x), 255]


class IndexView(TemplateView):
    template_name = "index.html"


class SingleImageObjectDetectionView(TemplateView):
    template_name = "object_detection/single_image.html"

    def get(self, request, *args, **kwargs):
        form = ClassifyImageForm()
        return self.render_to_response({
            "form": form,
        })

    def post(self, request, *args, **kwargs):
        form = ClassifyImageForm(request.POST, request.FILES)

        results = []
        if form.is_valid():
            cleaned = form.cleaned_data
            image_data = request.FILES['image_file'].read()
            image_np = cv2.imdecode(np.frombuffer(image_data, np.uint8), -1)
            min_confidence = cleaned['min_confidence'] / 100.
            annotations = classify_image_from_bytes(image_np)

            for name, annotation_group in annotations.items():
                annos = [a for a in annotation_group['results'] if a.score >= min_confidence]
                vis_image = visualize_annotation(image_np.copy(), annos)
                image_crops = crop_annotations(image_np, annos)
                results.append({
                    "visible_name": "{}... ({:.2f} seconds)".format(name[:16], annotation_group['runtime']),
                    "name": slugify(name),
                    "visualization": vis_image,
                    "annotations": [
                        {"score": a.score, "label": a.label, "crop_href": crop}
                        for a, crop in zip(annos, image_crops)]
                })

        return self.render_to_response({
            "form": form,
            "results": results
        })


class VerificationReportView(TemplateView):
    template_name = "object_detection/verification_report.html"

    def form_valid(self, request, form):
        cleaned = form.cleaned_data

        images = ExampleImage.objects

        labels = cleaned.get('labels')
        if labels:
            if not isinstance(labels, list):
                labels = [labels]
        else:
            labels = []
        if labels:
            images = ExampleImage.objects.filter(annotations__label__in=labels)

        source = cleaned.get('source')
        if source:
            images = images.filter(source=source)
        return images

    def handle_form(self, request, form):
        if form.is_valid():
            images = self.form_valid(request, form)
        else:
            images = ExampleImage.objects
        print(images.count())
        page = request.GET.get('page', 1)
        paginator = Paginator(images.all(), 25)
        print(paginator.get_page(page), page)
        images_page = paginator.get_page(page)

        return self.render_to_response({
            "form": form,
            "images_page": images_page
        })

    def get(self, request, *args, **kwargs):
        form = VerificationReportForm(request.GET)
        return self.handle_form(request, form)

    def post(self, request, *args, **kwargs):
        form = VerificationReportForm(request.POST, request.FILES)
        return self.handle_form(request, form)


class ExampleImageJsonView(View):

    def __get__(self, request, *args, **kwargs):
        request.GET.get("labels")
        return JsonResponse()
