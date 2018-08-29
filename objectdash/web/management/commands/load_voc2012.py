import os
import xml.etree.ElementTree as ET
from multiprocessing import cpu_count
from multiprocessing.pool import Pool

from django.core.files import File
from django.core.management.base import BaseCommand
from django.db import transaction

from objectdash.web.models import AnnotatedImage, Annotation, AnnotationLabel

from django import db
db.connections.close_all()


class Command(BaseCommand):
    help = 'Loads the VOC2012 (or a VOC compatible dataset) as ExampleImage instances'

    def add_arguments(self, parser):
        parser.add_argument('dataset_name')
        parser.add_argument('base_path')

    def load_annotations(self, annotation_dir):
        results = {}
        labels = set()
        for entry in os.scandir(annotation_dir):
            tree = ET.parse(entry.path)
            root = tree.getroot()

            objects = {}
            filename = root.find("filename").text
            for anno in root.iterfind("object"):
                label = anno.find("name").text
                labels.add(label)
                bndbox = anno.find("bndbox")
                xmin = int(float(bndbox.find('xmin').text))
                ymin = int(float(bndbox.find('ymin').text))
                xmax = int(float(bndbox.find('xmax').text))
                ymax = int(float(bndbox.find('ymax').text))
                objects.setdefault(label, []).append(
                    [xmin, ymin, xmax, ymax]
                )
            results[filename] = objects

        return results, labels

    def get_jpeg_paths(self, path):
        return {entry.name: entry.path
                for entry in os.scandir(path)
                if entry.name.lower().endswith('jpg')}

    def handle(self, *args, **options):
        dataset_name = options['dataset_name']
        base_path = options['base_path']
        annotations, labels = self.load_annotations(os.path.join(base_path, 'Annotations'))
        jpeg_images = self.get_jpeg_paths(os.path.join(base_path, 'JPEGImages'))

        label_objects = {}
        with transaction.atomic():
            for label in labels:
                label_objects[label], _ = AnnotationLabel.objects.get_or_create(label=label)

        instances = []
        with transaction.atomic():
            for name, path in jpeg_images.items():
                instance = AnnotatedImage()
                instance.source = dataset_name
                instance.image_file.save(name, File(open(path, 'rb')))
                instances.append(instance)

        with transaction.atomic():
            for instance, (name, path) in zip(instances, jpeg_images.items()):
                annos = annotations.get(name, {})
                for label, anno_instances in annos.items():
                    for xmin, ymin, xmax, ymax in anno_instances:
                        anno_inst = Annotation()
                        anno_inst.label = label_objects[label]
                        anno_inst.xmin = xmin
                        anno_inst.ymin = ymin
                        anno_inst.xmax = xmax
                        anno_inst.ymax = ymax
                        anno_inst.example_image = instance
                        anno_inst.save()
