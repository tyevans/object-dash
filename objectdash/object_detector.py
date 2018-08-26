import os

import cv2
import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import ops as utils_ops


class Rect:

    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1

        self.x2 = x2
        self.y2 = y2

    def translate(self, width, height):
        x1 = int(self.x1 * width)
        y1 = int(self.y1 * height)
        x2 = int(self.x2 * width)
        y2 = int(self.y2 * height)
        return x1, y1, x2, y2

    def crop(self, image_np):
        x1, y1, x2, y2 = self.translate(*image_np.shape[:2])
        return image_np[x1:x2, y1:y2]

    def draw(self, image_np, color, line_width=2):
        x1, y1, x2, y2 = self.translate(*image_np.shape[:2])
        cv2.rectangle(image_np, (y1, x1), (y2, x2), color, line_width)


class Mask:

    def __init__(self, mask):
        self.width, self.height = mask.shape[:2]
        self.mask = mask

    def draw(self, image_np, color):
        for s_row, m_row in zip(image_np, self.mask):
            for s_pixel, m_pixel in zip(s_row, m_row):
                if m_pixel == 1:
                    s_pixel[...] = np.average([s_pixel, color], axis=0)


class Annotation:

    def __init__(self, label, score, box):
        self.label = label
        self.rect = Rect(box[0], box[1], box[2], box[3])
        self.score = score

    def draw(self, image_np, color, draw_label=True):
        height, width = image_np.shape[:2]
        self.rect.draw(image_np, color)
        if draw_label:
            x = self.rect.x1 * width + 5
            y = self.rect.y1 * height + 5
            font = cv2.FONT_HERSHEY_SIMPLEX
            # cv2.putText(image_np, self.label, (x, y), font, 4, (255, 255, 255), 2, cv2.LINE_AA)

    @classmethod
    def from_results(cls, num_detections, labels, scores, boxes):
        annotations = []
        zipped = zip(labels[:num_detections], scores[:num_detections], boxes[:num_detections])
        for i, (label, score, box) in enumerate(zipped):
            annotations.append(cls(label, score, box))
        return annotations

    def crop(self, image_np):
        return self.rect.crop(image_np)


class MaskedAnnotation(Annotation):

    def __init__(self, label, score, box, mask):
        super(MaskedAnnotation, self).__init__(label, score, box)
        self.mask = Mask(mask)

    def draw(self, image_np, color, draw_label=True, draw_rect=False):
        self.mask.draw(image_np, color)

    @classmethod
    def from_results(cls, num_detections, labels, scores, boxes, masks):
        annotations = []
        zipped = zip(labels[:num_detections], scores[:num_detections], boxes[:num_detections], masks[:num_detections])
        for i, (label, score, box, mask) in enumerate(zipped):
            annotations.append(cls(label, score, box, mask))
        return annotations


class ObjectDetector:
    def __init__(self, graph_pb_file, label_file, num_classes):
        self.graph_pb_file = graph_pb_file
        self.label_file = label_file
        self.num_classes = num_classes
        self.label_map = label_map_util.load_labelmap(self.label_file)
        self.categories = label_map_util.convert_label_map_to_categories(
            self.label_map, max_num_classes=num_classes, use_display_name=True)
        self.category_index = label_map_util.create_category_index(self.categories)
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.graph_pb_file, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            self.session = tf.Session()

    @classmethod
    def load_dir(cls, graph_dir, num_classes):
        graph_pb_file = os.path.join(graph_dir, 'frozen_inference_graph.pb')
        label_file = os.path.join(graph_dir, 'labels.pbtxt')
        return cls(graph_pb_file, label_file, num_classes)

    def annotate(self, image_np):
        with self.detection_graph.as_default():
            # Get handles to input and output tensors
            graph = tf.get_default_graph()

            ops = graph.get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, image_np.shape[0], image_np.shape[1])
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed, 0)
            image_tensor = graph.get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = self.session.run(tensor_dict,
                                           feed_dict={image_tensor: np.expand_dims(image_np, 0)})

            num_detections = int(output_dict['num_detections'][0])
            classes = output_dict['detection_classes'][0].astype(np.uint8)
            labels = [self.category_index[x] for x in classes]
            boxes = output_dict['detection_boxes'][0].tolist()
            scores = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                masks = output_dict['detection_masks'][0]
                annotations = MaskedAnnotation.from_results(num_detections, labels, scores, boxes, masks)
            else:
                annotations = Annotation.from_results(num_detections, labels, scores, boxes)
        return annotations

    def close(self):
        self.session.close()

    def __del__(self):
        self.close()


class AggregateObjectDetector:

    def __init__(self, detectors=None):
        self.detectors = detectors or []

    @classmethod
    def load_dir(cls, models_dir, detector_cls=ObjectDetector):
        detectors = []
        for fileentry in os.scandir(models_dir):
            if fileentry.is_dir():
                obj_detector = detector_cls.load_dir(fileentry.path, 90)
                detectors.append(obj_detector)

        return cls(detectors)

    def iter_annotate(self, image_np):
        height, width = image_np.shape[:2]

        for detector in self.detectors:
            annotations = detector.annotate(image_np)
            yield annotations
