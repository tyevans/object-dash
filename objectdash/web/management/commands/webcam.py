import queue
import time
from multiprocessing import Process, Queue

import cv2
from django import db
from django.core.management.base import BaseCommand

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

    def run(self):
        detector = ObjectDetector(self.graph_file, self.label_file, self.num_classes)

        while True:
            frame = self.frame_queue.get()
            if frame is None:
                break
            annotations = detector.annotate(frame)
            self.annotation_queue.put(annotations)


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

    def handle(self, *args, **options):
        avi_out = options['avi_out']
        frame_queue = Queue()
        annotation_queue = Queue()
        processor = AnnotationProcessor(options['graph'], options['label_file'], options['num_classes'], frame_queue,
                                        annotation_queue)
        processor.start()

        cap = cv2.VideoCapture(0)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        if avi_out:
            out = cv2.VideoWriter(avi_out, fourcc, 20.0, (640, 480))

        ret, frame = cap.read()
        frame_queue.put(frame)

        count = 0
        last_time = 0
        annotations = None
        min_confidence = options['min_confidence']

        while (True):
            ret, frame = cap.read()
            annotations = self.get_annotations(annotation_queue, annotations, frame, frame_queue)
            if annotations is not None:
                self.draw_annotations(annotations, frame, min_confidence)
                if avi_out:
                    out.write(frame)

            if options['show_fps']:
                self.draw_fps(frame, last_time)

            cv2.imshow('frame', frame)

            try:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    frame_queue.put(None)
                    break
            except KeyboardInterrupt:
                frame_queue.put(None)
                raise
            count += 1

        cap.release()
        if avi_out:
            out.release()
        cv2.destroyAllWindows()

    def get_annotations(self, annotation_queue, annotations, frame, frame_queue):
        try:
            annotations = annotation_queue.get_nowait()
        except queue.Empty:
            pass
        else:
            frame_queue.put(frame)
        return annotations

    def draw_annotations(self, annotations, frame, min_confidence):
        for i, annotation in enumerate(annotations):
            if annotation.score >= min_confidence:
                annotation.draw(frame, (0, 255, 0))

    def draw_fps(self, frame, last_time):
        font = cv2.FONT_HERSHEY_PLAIN
        fontScale = 1
        lineType = 2
        cv2.putText(frame, str(int(1 / (time.time() - self._last_time))) + " FPS", (10, 40), font,
                    fontScale,
                    (0, 0, 255),
                    lineType)
        self._last_time = time.time()
