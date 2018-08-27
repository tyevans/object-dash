from django.db import models

from objectdash.detect.object_detector import ObjectDetector


class ObjectDetectorClosed(Exception):
    pass


_object_detectors = {}


def object_detector_upload_to(instance, filename):
    return 'object_detectors/pb/{0}/{1}'.format(instance.name, filename)


class PBObjectDetector(models.Model):
    id = models.UUIDField(primary_key=True)
    name = models.TextField(null=False)
    pb_file = models.FileField(help_text="frozen_inference_graph.pb", upload_to=object_detector_upload_to, null=False)
    label_file = models.FileField(help_text="labels.pbtxt", upload_to=object_detector_upload_to, null=False)
    num_classes = models.IntegerField(null=False, default=90)
    active = models.BooleanField(default=True, null=False)

    class Meta:
        app_label = "web"
        verbose_name = "Object Detector"
        verbose_name_plural = "Object Detectors"

    def get_detector(self):
        global _object_detectors
        od = _object_detectors.get(self.name)
        if od is None:
            od = _object_detectors[self.name] = ObjectDetector(self.pb_file.path, self.label_file.path,
                                                               self.num_classes)
        return od

    def annotate(self, image_np):
        if self.active:
            return self.get_detector().annotate(image_np)
        else:
            raise ObjectDetectorClosed

    def close(self):
        return self.get_detector().close()

    def save(self, force_insert=False, force_update=False, using=None, update_fields=None):
        global _object_detectors
        if self.name in _object_detectors and self.active is False:
            print("Tearing down", self.name)
            self.close()
            del _object_detectors[self.name]
        elif self.name not in _object_detectors and self.active is True:
            pass

        super().save(force_insert, force_update, using, update_fields)

    def __str__(self):
        return self.name
