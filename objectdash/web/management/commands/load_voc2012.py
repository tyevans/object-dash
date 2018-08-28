import os
import xml.etree.ElementTree as ET
from multiprocessing import cpu_count
from multiprocessing.pool import Pool

from django.core.files import File
from django.core.management.base import BaseCommand
from django.db import transaction

from objectdash.web.models import ExampleImage, ExampleAnnotation

from django import db
db.connections.close_all()


class Command(BaseCommand):
    help = 'Loads the VOC2012 (or a VOC compatible dataset) as ExampleImage instances'

    def add_arguments(self, parser):
        parser.add_argument('dataset_name')
        parser.add_argument('base_path')

    def load_annotations(self, annotation_dir):
        results = {}
        for entry in os.scandir(annotation_dir):
            tree = ET.parse(entry.path)
            root = tree.getroot()

            objects = {}
            filename = root.find("filename").text
            for anno in root.iterfind("object"):
                label = anno.find("name").text
                bndbox = anno.find("bndbox")
                xmin = int(float(bndbox.find('xmin').text))
                ymin = int(float(bndbox.find('ymin').text))
                xmax = int(float(bndbox.find('xmax').text))
                ymax = int(float(bndbox.find('ymax').text))
                objects.setdefault(label, []).append(
                    [xmin, ymin, xmax, ymax]
                )
            results[filename] = objects

        return results

    def get_jpeg_paths(self, path):
        return {entry.name: entry.path
                for entry in os.scandir(path)
                if entry.name.lower().endswith('jpg')}

    def handle(self, *args, **options):
        dataset_name = options['dataset_name']
        base_path = options['base_path']
        annotations = self.load_annotations(os.path.join(base_path, 'Annotations'))
        jpeg_images = self.get_jpeg_paths(os.path.join(base_path, 'JPEGImages'))

        instances = []
        with transaction.atomic():
            for name, path in jpeg_images.items():
                instance = ExampleImage()
                instance.source = dataset_name
                instance.image_file.save(name, File(open(path, 'rb')))
                instances.append(instance)

        with transaction.atomic():
            for instance, (name, path) in zip(instances, jpeg_images.items()):
                annos = annotations.get(name, {})
                for label, anno_instances in annos.items():
                    for xmin, ymin, xmax, ymax in anno_instances:
                        anno_inst = ExampleAnnotation()
                        anno_inst.label = label
                        anno_inst.xmin = xmin
                        anno_inst.ymin = ymin
                        anno_inst.xmax = xmax
                        anno_inst.ymax = ymax
                        anno_inst.example_image = instance
                        anno_inst.save()
