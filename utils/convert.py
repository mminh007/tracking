


def convert_yolo_to_xy(yolo_line, img_width, img_height):
    """
    Converts a YOLO format bounding box to xmin, ymin, xmax, ymax.

    Args:
        yolo_line: A string representing a YOLO format bounding box.
        img_width: The width of the image.
        img_height: The height of the image.

    Returns:
        A tuple containing (xmin, ymin, xmax, ymax).
    """

    class_id, x_center, y_center, width, height = map(float, yolo_line.split())
    xmin = int((x_center - width / 2) * img_width)
    ymin = int((y_center - height / 2) * img_height)
    xmax = int((x_center + width / 2) * img_width)
    ymax = int((y_center + height / 2) * img_height)

    return xmin, ymin, xmax, ymax, class_id


def convert_yolo_to_coco(yolo_line, img_width, img_height):
    class_id, x_center, y_center, width, height = map(float, yolo_line.split())
    x_min = int((x_center - width / 2) * width)
    y_min = int((y_center - height / 2) * height)

    box_width = int(width * img_width) 
    box_height = int(height * img_height)

    return x_min, y_min, box_width, box_height, class_id
