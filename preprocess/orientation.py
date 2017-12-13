import cv2.aruco as aruco
import cv2
import numpy as np
from PIL import Image
from functools import partial

_dictionary = aruco.getPredefinedDictionary(aruco.DICT_ARUCO_ORIGINAL)

_destination_dimensions = np.array([
    [0, 0],
    [4000, 0],
    [4000, 3000],
    [0, 3000],
], dtype = 'float32')

def correct_orientation(image):
    marker_boxes, identifiers, _ = aruco.detectMarkers(image, _dictionary)
    assert len(marker_boxes) == 4, 'one of the aruco markers cannot be found'

    marker_boxes_array = np.array(marker_boxes).squeeze()

    middle = marker_boxes_array.mean(axis = 1).mean(axis = 0)
    extreme_corners = np.array([
        _extreme(middle, marker_box) for marker_box in marker_boxes_array
    ])
    points = { identifier: corner for identifier, corner in zip(list(identifiers.squeeze()), extreme_corners) }

    corners_to_transform = np.array([
        points[0],
        points[1],
        points[2],
        points[3]
    ], dtype = 'float32')

    M = cv2.getPerspectiveTransform(corners_to_transform, _destination_dimensions)
    return cv2.warpPerspective(image, M, (4000, 3000))


def _extreme(middle, box):
    """
    Find the corner of a box that's most extreme with respect to some
    centre point `middle`.
    """
    xs = box[:, 0]
    ys = box[:, 1]
    distances = np.sqrt((xs - middle[0]) ** 2 + (ys - middle[1]) ** 2)
    sorting = np.argsort(-distances)
    return box[sorting][0]
