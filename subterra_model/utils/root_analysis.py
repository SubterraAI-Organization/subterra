import cv2
import numpy as np
from skimage.morphology import skeletonize


def find_root_count(image: np.ndarray) -> int:
    image_contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
    return len(image_contours)


def find_total_root_length(image: np.ndarray, scaling_factor: float) -> float:
    skeleton = skeletonize(image)
    return float(np.sum(skeleton) * scaling_factor)


def find_total_root_area(image: np.ndarray, scaling_factor: float) -> float:
    return float(np.sum(image / 255) * (scaling_factor**2))


def find_root_diameter(image: np.ndarray, scaling_factor: float) -> float:
    skeleton = skeletonize(image).astype(np.uint8)
    image_contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    y, x = np.where(skeleton == 1)
    if len(x) == 0 or len(y) == 0:
        return 0.0

    image_contour_points = np.vstack(image_contours).squeeze(axis=1)

    diameters = []
    for point in zip(x, y):
        point = np.array(point)[np.newaxis, :]
        distances = np.linalg.norm(image_contour_points - point, axis=1)
        diameters.append(2 * np.min(distances))

    return float(np.mean(diameters) * scaling_factor)


def find_total_root_volume(image: np.ndarray, scaling_factor: float) -> float:
    skeleton = skeletonize(image).astype(np.uint8)
    image_contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    y, x = np.where(skeleton == 1)
    if len(x) == 0 or len(y) == 0:
        return 0.0

    image_contour_points = np.vstack(image_contours).squeeze()

    radii = []
    for point in zip(x, y):
        point = np.array(point)[np.newaxis, :]
        distances = np.linalg.norm(image_contour_points - point, axis=1)
        radii.append(np.min(distances) * scaling_factor)

    return float(np.sum(np.pi * (np.array(radii)**2)))


def calculate_metrics(image: np.ndarray, scaling_factor: float) -> dict:
    root_count = find_root_count(image)
    if root_count == 0:
        return {
            "root_count": 0,
            "average_root_diameter": 0.0,
            "total_root_length": 0.0,
            "total_root_area": 0.0,
            "total_root_volume": 0.0,
        }

    return {
        "root_count": root_count,
        "average_root_diameter": find_root_diameter(image, scaling_factor),
        "total_root_length": find_total_root_length(image, scaling_factor),
        "total_root_area": find_total_root_area(image, scaling_factor),
        "total_root_volume": find_total_root_volume(image, scaling_factor),
    }

