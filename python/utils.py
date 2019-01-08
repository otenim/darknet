import numpy as np
import cv2


def draw_bounding_boxes(
    img, boxes, ssd_size,
    classes, rgb_color_map=None):

    if len(img.shape) != 3:
        raise ValueError(
            'img is expected to have the following 3D tensor shape: '
            '(img_height, img_width, img_channels), but this time '
            'len(img.shape) = %d' % len(img.shape)
        )

    if len(ssd_size) != 2:
        raise ValueError(
            'ssd_size is expected to have two elements '
            '(input_height, input_width), '
            'but this time len(ssd_size) = %d' % len(ssd_size)
        )

    if rgb_color_map:
        if len(rgb_color_map) != len(classes):
            raise ValueError(
                'rgb_color_map and classes must have the same length of '
                'elements, but this time len(rgb_color_map) = %d, '
                'while len(classes) = %d' % (len(rgb_color_map), len(classes))
            )
    else:
        rgb_color_map = []
        for i in range(len(classes)):
            color = (
                np.random.randint(0,255),
                np.random.randint(0,255),
                np.random.randint(0,255)
            )
            rgb_color_map.append(color)

    img_out = img.copy()
    if len(boxes) == 0:
        return img_out

    img_h, img_w, _ = img.shape
    input_h, input_w = ssd_size

    for box in boxes:
        # box[0]: class_id
        # box[1]: confidence score
        # box[2],box[3]: xmin,ymin
        # box[5],box[4]: xmax,ymax
        class_id = int(box[0])
        conf = box[1]
        xmin = max(0, int(img_w * box[2] / input_w))
        ymin = max(0, int(img_h * box[3] / input_h))
        xmax = min(img_w, int(img_w * box[4] / input_w))
        ymax = min(img_h, int(img_h * box[5] / input_h))
        rgb_color = rgb_color_map[class_id]
        box_label = '%s: %.2f' % (classes[class_id], conf)

        # draw the detected box on img_out
        cv2.rectangle(
            img_out,
            (xmin,ymin),(xmax,ymax),
            color=rgb_color,
            thickness=2,
            lineType=cv2.LINE_AA
        )

        # paste the box's class name on img_out
        font_size = 12
        font = cv2.FONT_HERSHEY_SIMPLEX
        label_w = font_size * (len(box_label))
        label_h = font_size
        cv2.rectangle( # background
            img_out,
            (xmin,ymin),
            (xmin+label_w,ymin+label_h),
            color=rgb_color,
            thickness=-1,
            lineType=cv2.LINE_AA,
        )
        cv2.rectangle( # edge
            img_out,
            (xmin,ymin),
            (xmin+label_w,ymin+label_h),
            color=(0,0,0),
            thickness=1,
            lineType=cv2.LINE_AA
        )
        cv2.putText( # text
            img_out,
            box_label,
            (xmin+3,ymin+label_h-3),
            font,
            0.40,
            (255,255,255),
            lineType=cv2.LINE_AA)
    return img_out
