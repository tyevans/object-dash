import queue
import time
from multiprocessing import Process, Queue

import cv2
from django import db
from django.core.management.base import BaseCommand

from objectdash.detect.object_detector import ObjectDetector

db.connections.close_all()

outdir = "/home/baddad/workspace/object-dash/media/crops"


class AnnotationProcessor(Process):

    def __init__(self, frame_queue, annotation_queue, **kwargs):
        super(AnnotationProcessor, self).__init__()
        self.frame_queue = frame_queue
        self.annotation_queue = annotation_queue
        self.kwargs = kwargs

    def run(self):
        detector = ObjectDetector(
            "/home/baddad/workspace/object-dash/data/models/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/frozen_inference_graph.pb",
            "./media/object_detectors/pb/ssd_resnet50_v1_fpn/labels.pbtxt",
            90
        )

        while True:
            frame = self.frame_queue.get()
            if frame is None:
                break
            annotations = detector.annotate(frame)
            self.annotation_queue.put(annotations)


class Command(BaseCommand):
    help = ''

    def handle(self, *args, **kwargs):
        frame_queue = Queue()
        annotation_queue = Queue()
        processor = AnnotationProcessor(frame_queue, annotation_queue)
        processor.start()

        cap = cv2.VideoCapture(0)

        ret, frame = cap.read()
        frame_queue.put(frame)

        count = 0
        last_time = time.time()
        annotations = None

        while (True):
            ret, frame = cap.read()
            annotations = self.get_annotations(annotation_queue, annotations, frame, frame_queue)
            if annotations is not None:
                self.draw_annotations(annotations, frame)

            self.draw_fps(frame, last_time)
            cv2.imshow('frame', frame)

            try:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    frame_queue.put(None)
                    break
            except KeyboardInterrupt:
                frame_queue.put(None)
                raise

            last_time = time.time()
            count += 1

        cap.release()
        cv2.destroyAllWindows()

    def get_annotations(self, annotation_queue, annotations, frame, frame_queue):
        try:
            annotations = annotation_queue.get_nowait()
        except queue.Empty:
            pass
        else:
            frame_queue.put(frame)
        return annotations

    def draw_annotations(self, annotations, frame):
        for i, annotation in enumerate(annotations):
            if annotation.score >= 0.5:
                annotation.draw(frame, (0, 255, 0))

    def draw_fps(self, frame, last_time):
        font = cv2.FONT_HERSHEY_PLAIN
        fontScale = 1
        lineType = 2
        cv2.putText(frame, str(int((time.time() - last_time) * 1000)) + " FPS", (10, 40), font,
                    fontScale,
                    (0, 0, 255),
                    lineType)
