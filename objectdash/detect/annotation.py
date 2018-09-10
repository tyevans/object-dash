import cv2
import numpy as np


class Rect:

    def __init__(self, y1, x1, y2, x2):
        self.x1 = x1
        self.y1 = y1

        self.x2 = x2
        self.y2 = y2

    def translate(self, height, width):
        x1 = int(self.x1 * width)
        y1 = int(self.y1 * height)
        x2 = int(self.x2 * width)
        y2 = int(self.y2 * height)
        return y1, x1, y2, x2

    def crop(self, image_np):
        y1, x1, y2, x2 = self.translate(*image_np.shape[:2])
        return image_np[y1:y2, x1:x2]

    def draw(self, image_np, color, line_width=2):
        y1, x1, y2, x2 = self.translate(*image_np.shape[:2])
        cv2.rectangle(image_np, (x1, y1), (x2, y2), color, line_width)

    def __str__(self):
        return "{}, {}, {}, {}".format(self.x1, self.y1, self.x2 ,self.y2)

class Mask:

    def __init__(self, mask, one_based=True):
        self.width, self.height = mask.shape[:2]
        self.mask = mask
        self.one_based = one_based

    def as_transparency(self, image_np):
        height, width= image_np.shape[:2]
        output = np.zeros([height, width, 4], dtype=np.uint8)
        for o_row, s_row, m_row in zip(output, image_np, self.mask):
            for o_pixel, s_pixel, m_pixel in zip(o_row, s_row, m_row):
                if m_pixel == 1 or m_pixel == 255:
                    o_pixel[...] = np.array([s_pixel[0], s_pixel[1], s_pixel[2], 255], dtype=np.uint8)
        return output

    def apply(self, image_np):
        if self.one_based:
            mask = self.mask * 255
        else:
            mask = self.mask
        return cv2.bitwise_and(image_np, image_np, mask = mask)

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
            font = cv2.FONT_HERSHEY_PLAIN
            fontScale = 1
            lineType = 1
            padding = 5
            d_padding = padding * 2
            label = "{} ({:.2f})".format(self.label['name'], self.score)
            text_dims, _ = cv2.getTextSize(label, font, fontScale, lineType)
            text_width, text_height = text_dims
            x = int(self.rect.x1 * width)
            y = int(self.rect.y1 * height)
            cv2.rectangle(image_np, (x, y - text_height - d_padding), (x + text_width + d_padding, y), color, -1)
            cv2.putText(image_np, label, (x + padding, y - padding), font, fontScale, (0, 0, 0), lineType)

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

    def __init__(self, label, score, box, mask=None):
        super(MaskedAnnotation, self).__init__(label, score, box)
        self.mask = Mask(mask if mask is not None else [])

    def crop(self, image_np):
        transparent = self.mask.as_transparency(image_np)
        return self.rect.crop(transparent)

    def draw(self, image_np, color, draw_label=True, draw_rect=False):
        self.mask.draw(image_np, color)

    @classmethod
    def from_results(cls, num_detections, labels, scores, boxes, masks=None):
        masks = masks if masks is not None else []
        annotations = []
        zipped = zip(labels[:num_detections], scores[:num_detections], boxes[:num_detections], masks[:num_detections])
        for i, (label, score, box, mask) in enumerate(zipped):
            annotations.append(cls(label, score, box, mask))
        return annotations
