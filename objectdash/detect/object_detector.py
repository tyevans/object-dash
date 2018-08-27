import os

import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import ops as utils_ops

from objectdash.detect.annotation import Annotation, MaskedAnnotation


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
        for detector in self.detectors:
            annotations = detector.annotate(image_np)
            yield annotations
