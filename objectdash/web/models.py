import os
import uuid

import tensorflow as tf
from django.db import models
from object_detection.utils import dataset_util
from object_detection import model_hparams
from object_detection import model_lib

from objectdash.detect.object_detector import ObjectDetector as TFObjectDetector


class ObjectDetectorClosed(Exception):
    pass


_object_detectors = {}


def object_detector_upload_to(instance, filename):
    return 'object_detectors/pb/{0}/{1}'.format(instance.name, filename)


def example_image_upload_to(instance, filename):
    base, ext = os.path.splitext(filename)
    id = str(uuid.uuid4())
    a, b, c, = list(str(id)[:3])
    return 'example_images/{}/{}/{}/{}{}'.format(a, b, c, id, ext)


class UntrainableModelException(Exception):
    pass


class ObjectDetectionModel(models.Model):
    id = models.AutoField(primary_key=True)
    name = models.TextField(null=False)
    pb_file = models.FileField(help_text="frozen_inference_graph.pb", upload_to=object_detector_upload_to, null=False)
    label_file = models.FileField(help_text="labels.pbtxt", upload_to=object_detector_upload_to, null=False)
    ckpt_data_file = models.FileField(null=True, upload_to=object_detector_upload_to)
    ckpt_index_file = models.FileField(null=True, upload_to=object_detector_upload_to)
    ckpt_meta_file = models.FileField(null=True, upload_to=object_detector_upload_to)
    pipeline_config_template = models.FileField(null=True, upload_to=object_detector_upload_to)
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
            od = _object_detectors[self.name] = TFObjectDetector(self.pb_file.path, self.label_file.path,
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
            self.close()
            del _object_detectors[self.name]
        elif self.name not in _object_detectors and self.active is True:
            pass

        super().save(force_insert, force_update, using, update_fields)

    def train_new_model(self, image_set, hparam_overrides='', num_train_steps=100, num_eval_steps=30):
        if any(x is None for x in [self.ckpt_data_file, self.ckpt_index_file, self.ckpt_meta_file]):
            raise UntrainableModelException

        model_dir = object_detector_upload_to(self, '')
        config = tf.estimator.RunConfig(model_dir=model_dir)
        train_and_eval_dict = model_lib.create_estimator_and_inputs(
            run_config=config,
            hparams=model_hparams.create_hparams(hparam_overrides),
            pipeline_config_path=FLAGS.pipeline_config_path,
            train_steps=num_train_steps,
            eval_steps=num_eval_steps)
        estimator = train_and_eval_dict['estimator']
        train_input_fn = train_and_eval_dict['train_input_fn']
        eval_input_fn = train_and_eval_dict['eval_input_fn']
        eval_on_train_input_fn = train_and_eval_dict['eval_on_train_input_fn']
        predict_input_fn = train_and_eval_dict['predict_input_fn']
        train_steps = train_and_eval_dict['train_steps']
        eval_steps = train_and_eval_dict['eval_steps']

    def __str__(self):
        return self.name


class AnnotatedImage(models.Model):
    class Meta:
        ordering = ['id']

    id = models.AutoField(primary_key=True)
    image_file = models.ImageField(upload_to=example_image_upload_to, height_field="height", width_field="width")
    width = models.IntegerField()
    height = models.IntegerField()
    source = models.TextField()

    def tf_example(self):
        _, ext = os.path.split(self.image_file.name)
        image_format = ext[1:].lower()

        xmins, ymins, xmaxs, ymaxs = [], [], [], []
        classes_text = []
        classes = []

        annotations = self.annotations.all()
        for annotation in annotations:
            xmins.append(annotation.xmin / self.width)
            ymins.append(annotation.ymin / self.height)
            xmaxs.append(annotation.xmax / self.width)
            ymaxs.append(annotation.ymax / self.height)
            classes_text.append(bytes(annotation.label.label, 'utf-8'))
            classes.append(annotation.label.id)

        tf_example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': dataset_util.int64_feature(self.height),
            'image/width': dataset_util.int64_feature(self.width),
            'image/filename': dataset_util.bytes_feature(bytes(self.image_file.name, 'utf-8')),
            'image/source_id': dataset_util.bytes_feature(bytes(self.image_file.name, 'utf-8')),
            'image/encoded': dataset_util.bytes_feature(open(self.image_file.path, 'rb').read()),
            'image/format': dataset_util.bytes_feature(bytes(image_format, 'utf-8')),
            'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
            'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
            'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
            'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
            'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
            'image/object/class/label': dataset_util.int64_list_feature(classes),
        }))
        return tf_example


class AnnotationLabel(models.Model):
    class Meta:
        app_label = "web"
        verbose_name = "Annotation Label"
        verbose_name_plural = "Annotation Labels"

    id = models.AutoField(primary_key=True)
    label = models.TextField(null=False)


class Annotation(models.Model):
    class Meta:
        app_label = "web"
        verbose_name = "Annotation"
        verbose_name_plural = "Annotations"

    id = models.AutoField(primary_key=True)
    label = models.ForeignKey(AnnotationLabel, null=False, on_delete=models.CASCADE)
    example_image = models.ForeignKey(AnnotatedImage, related_name="annotations", on_delete=models.CASCADE, null=False)
    xmin = models.IntegerField(null=False)
    xmax = models.IntegerField(null=False)
    ymin = models.IntegerField(null=False)
    ymax = models.IntegerField(null=False)


def tf_record_file_upload_to(instance, filename):
    base, ext = os.path.splitext(filename)
    id = str(uuid.uuid4())
    a, b, c, = list(str(id)[:3])
    return 'tf_records/{}/{}/{}/{}{}'.format(a, b, c, id, ext)


class TFRecord(models.Model):
    class Meta:
        app_label = "web"
        verbose_name = "Tensorflow Record"
        verbose_name_plural = "Tensorflow Records"

    id = models.AutoField(primary_key=True)
    tf_record_file = models.FileField(upload_to=tf_record_file_upload_to)

