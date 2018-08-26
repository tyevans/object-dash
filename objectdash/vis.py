def draw_annotations(image_np, annotations):
    image = image_np.copy()
    for i, box in annotations['detection_boxes']:
        normalize_box()
