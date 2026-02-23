import json

import cv2
import numpy as np


def to_labelme(image_filename: str, image: np.ndarray) -> str:
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
    shapes = []
    for contour in contours:
        points = contour.squeeze(1).tolist()
        if len(points) < 3:
            continue
        shapes.append({
            "label": "root",
            "points": points,
            "group_id": None,
            "shape_type": "polygon",
            "flags": {},
        })

    return json.dumps({
        "version": "4.6.0",
        "flags": {},
        "shapes": shapes,
        "imagePath": image_filename,
        "imageData": None,
        "imageHeight": image.shape[0],
        "imageWidth": image.shape[1],
    })


def threshold(mask: np.ndarray, threshold_area: int = 50) -> np.ndarray:
    output_contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if len(output_contours) == 0:
        return mask

    hierarchy = hierarchy.squeeze(0)

    threshold_contours = []
    threshold_hierarchy = []
    for i in range(len(output_contours)):
        if hierarchy[i][3] != -1:
            continue

        current_index = hierarchy[i][2]
        contour_area = cv2.contourArea(output_contours[i])
        while current_index != -1:
            contour_area -= cv2.contourArea(output_contours[current_index])
            current_index = hierarchy[current_index][0]

        if contour_area < threshold_area:
            continue

        threshold_contours.append(output_contours[i])
        threshold_hierarchy.append(hierarchy[i])

        current_index = hierarchy[i][2]
        while current_index != -1:
            threshold_contours.append(output_contours[current_index])
            threshold_hierarchy.append(hierarchy[current_index])
            current_index = hierarchy[current_index][0]

    thresholded_mask = np.zeros(mask.shape, dtype=np.uint8)

    for i in range(len(threshold_contours)):
        if threshold_hierarchy[i][3] != -1:
            continue
        cv2.drawContours(thresholded_mask, threshold_contours, i, 255, cv2.FILLED)

    for i in range(len(threshold_contours)):
        if threshold_hierarchy[i][3] == -1:
            continue
        cv2.drawContours(thresholded_mask, threshold_contours, i, 0, cv2.FILLED)

    return thresholded_mask

