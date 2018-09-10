from __future__ import division

import os
import queue
import time
from multiprocessing import Process, Queue

import cv2
import numpy as np
from django import db
from django.core.management.base import BaseCommand

from objectdash.detect.annotation import Mask
from objectdash.detect.object_detector import ObjectDetector

db.connections.close_all()


class AnnotationProcessor(Process):

    def __init__(self, graph_file, label_file, num_classes, frame_queue, annotation_queue, **kwargs):
        super(AnnotationProcessor, self).__init__()
        self.graph_file = graph_file
        self.label_file = label_file
        self.num_classes = num_classes
        self.frame_queue = frame_queue
        self.annotation_queue = annotation_queue
        self.anno_frame = None

    def run(self):
        detector = ObjectDetector(self.graph_file, self.label_file, self.num_classes)

        while True:
            frame = self.frame_queue.get()
            if frame is None:
                break
            annotations = detector.annotate(frame)
            self.annotation_queue.put(annotations)


class ImageHandler(object):

    def apply_first(self, ret, image_np):
        return image_np

    def apply(self, ret, image_np):
        return image_np

    def close(self):
        pass


class BackgroundSubtractor(ImageHandler):

    def __init__(self):
        self.annotations = None
        self.fgbg = cv2.createBackgroundSubtractorMOG2()

    def apply_first(self, ret, image_np):
        return self.apply(ret, image_np)

    def apply(self, ret, image_np):
        self.fgbg.apply(image_np)
        motion_mask = self.fgbg.apply(image_np)
        self.motion_mask = Mask(motion_mask)
        return self.motion_mask.apply(image_np)


class TrackingAnnotation(object):

    def __init__(self, image_np, annotation):
        self.image_np = image_np
        self.annotation = annotation
        self.height, self.width = self.image_np.shape[:2]
        y1, x1, y2, x2 = self.annotation.rect.translate(self.height, self.width)
        self.track_window = (x1, y1, x2 - x1, y2 - y1)

        roi = image_np[y1:y2, x1:x2]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
        self.roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
        cv2.normalize(self.roi_hist, self.roi_hist, 0, 255, cv2.NORM_MINMAX)
        self.term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 2)

    def step(self, image_np):
        hsv = cv2.cvtColor(image_np, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0], self.roi_hist, [0, 180], 1)

        ret, self.track_window = cv2.meanShift(dst, self.track_window, self.term_crit)
        x, y, width, height = self.track_window

        self.annotation.rect.x1 = x / self.width
        self.annotation.rect.x2 = (x + width) / self.width

        self.annotation.rect.y1 = y / self.height
        self.annotation.rect.y2 = (y + height) / self.height

    def __getattr__(self, item):
        return getattr(self.annotation, item)


class ImageAnnotator(ImageHandler):

    def __init__(self, graph, label_file, num_class, crop_dir=None, min_confidence=0.5):
        self.crop_dir = crop_dir
        self.min_confidence = min_confidence
        self.frame_queue = Queue()
        self.annotation_queue = Queue()
        self.processor = AnnotationProcessor(graph, label_file, num_class, self.frame_queue, self.annotation_queue)
        self.processor.start()
        self.annotations = []

    def apply_first(self, ret, image_np):
        self.anno_frame = image_np.copy()
        self.frame_queue.put(image_np)
        self.last_annotation = time.time()
        return image_np

    def apply(self, ret, image_np):
        try:
            self.annotations = self.annotation_queue.get_nowait()
            # self.annotations = [a for a in self.annotations if a.label['name'] == 'person']
        except queue.Empty:
            if ret:
                for annotation in self.annotations:
                    annotation.step(image_np)
        else:
            self.annotations = [TrackingAnnotation(self.anno_frame, anno) for anno in self.annotations]
            if self.crop_dir:
                self.crop_annotations()
            self.anno_frame = image_np.copy()
            self.frame_queue.put(image_np)

        self.draw_annotations(image_np)

        return image_np

    def close(self):
        self.frame_queue.put(None)

    def crop_annotations(self):
        frame_time = time.time()
        height, width = self.anno_frame.shape[:2]
        for i, annotation in enumerate(self.annotations):
            y1, x1, y2, x2 = annotation.rect.translate(height, width)
            anno_height = y2 - y1
            anno_width = x2 - x1
            if annotation.score >= self.min_confidence and anno_height * anno_width >= 400:
                label = "{}_{}.jpg".format(annotation.label['name'], int(frame_time))
                path = os.path.join(self.crop_dir, label)
                cv2.imwrite(path, annotation.crop(self.anno_frame))

    def draw_annotations(self, image_np):
        for i, annotation in enumerate(self.annotations):
            if annotation.score >= self.min_confidence:
                annotation.draw(image_np, (0, 255, 0))


class AVIOutput(ImageHandler):

    def __init__(self, output_path):
        self.output_path = output_path
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self._output = cv2.VideoWriter(output_path, fourcc, 20.0, (640, 480))

    def apply_first(self, ret, image_np):
        return self.apply(ret, image_np)

    def apply(self, ret, image_np):
        if ret:
            self._output.write(image_np)
        return image_np

    def close(self):
        self._output.release()


class FPSCounter(ImageHandler):

    def __init__(self):
        self.last_time = time.time()

    def apply_first(self, ret, image_np):
        return self.apply(ret, image_np)

    def apply(self, ret, image_np):
        font = cv2.FONT_HERSHEY_PLAIN
        fontScale = 1
        lineType = 2
        cv2.putText(image_np, str(int(1 / (time.time() - self.last_time))) + " FPS", (10, 40), font,
                    fontScale,
                    (0, 0, 255),
                    lineType)
        self.last_time = time.time()
        return image_np


class Command(BaseCommand):
    help = ''

    def __init__(self):
        super(Command, self).__init__()
        self._last_time = 0

    def add_arguments(self, parser):

        parser.add_argument('graph')
        parser.add_argument('label_file')
        parser.add_argument('num_classes', type=int)
        parser.add_argument('--min_confidence', type=float, default=0.5)
        parser.add_argument('--show_fps', action="store_true", default=False)
        parser.add_argument('--avi_out', default=None)
        parser.add_argument('--crop_dir', default=None)
        parser.add_argument('--subtract_bg', action="store_true", default=False)

    def setup_pipeline(self, options):
        pipeline = []

        if options['subtract_bg']:
            pipeline.append(BackgroundSubtractor())

        pipeline.append(
            ImageAnnotator(options['graph'], options['label_file'], options['num_classes'],
                           options['crop_dir'], options['min_confidence'])
        )

        if options['show_fps']:
            pipeline.append(FPSCounter())

        if options['avi_out']:
            pipeline.append(AVIOutput(options['avi_out']))

        return pipeline

    def handle(self, *args, **options):

        cap = cv2.VideoCapture(0)
        pipeline = self.setup_pipeline(options)

        ret, frame = cap.read()
        for handler in pipeline:
            frame = handler.apply_first(ret, frame)

        while (True):
            ret, frame = cap.read()

            for handler in pipeline:
                frame = handler.apply(ret, frame)

            cv2.imshow('frame', frame)
            try:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    for handler in pipeline:
                        handler.close()
                    break
            except KeyboardInterrupt:
                for handler in pipeline:
                    handler.close()
                raise

        cap.release()
        cv2.destroyAllWindows()
