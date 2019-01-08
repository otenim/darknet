import numpy as np
import cv2


def draw_bounding_boxes(
    img, bboxes, color_map):

    img_out = img.copy()
    if len(bboxes) == 0:
        return img_out

    img_h, img_w, _ = img.shape

    for bbox in bboxes:
        cname, conf, (xc, yc, w, h) = bbox
        cname = cname.decode()
        xmin = int(xc - w / 2)
        ymin = int(yc - h / 2)
        xmax = int(xc + w / 2)
        ymax = int(yc + h / 2)
        color = color_map[cname]
        label = '%s: %.2f' % (cname, conf)

        # draw the detected box on img_out
        cv2.rectangle(
            img_out,
            (xmin, ymin), (xmax, ymax),
            color=color,
            thickness=2,
            lineType=cv2.LINE_AA
        )

        # paste the box's class name on img_out
        font_size = 12
        font = cv2.FONT_HERSHEY_SIMPLEX
        label_w = font_size * (len(label))
        label_h = font_size
        cv2.rectangle( # background
            img_out,
            (xmin, ymin),
            (xmin + label_w, ymin + label_h),
            color=color,
            thickness=-1,
            lineType=cv2.LINE_AA,
        )
        cv2.rectangle( # edge
            img_out,
            (xmin, ymin),
            (xmin + label_w, ymin + label_h),
            color=(0, 0, 0),
            thickness=1,
            lineType=cv2.LINE_AA
        )
        cv2.putText( # text
            img_out,
            label,
            (xmin + 3, ymin + label_h - 3),
            font,
            0.40,
            (255,255,255),
            lineType=cv2.LINE_AA)
    return img_out
