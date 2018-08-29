import time

import cv2
import numpy as np
from django.contrib.staticfiles.templatetags.staticfiles import static
from django.utils.text import slugify
from django.views.generic import TemplateView

from objectdash.visualize import visualize_annotation, crop_annotations
from objectdash.web.forms import ClassifyImageForm
from objectdash.web.models import PBObjectDetector


def classify_image_from_bytes(image_np):
    detectors = PBObjectDetector.objects.filter(active=True).all()
    annotations = {}

    if image_np.shape[2] == 4:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGRA2BGR)

    for detector in detectors:
        start_time = time.time()
        annotations[detector.name] = {
            "results": detector.annotate(image_np)
        }
        annotations[detector.name]['runtime'] = time.time() - start_time
    return annotations


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
                vis_image = static(visualize_annotation(image_np.copy(), annos))
                image_crops = [static(x) for x in crop_annotations(image_np, annos)]
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
