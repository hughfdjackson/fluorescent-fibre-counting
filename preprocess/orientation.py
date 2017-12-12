import cv2.aruco as aruco
import cv2
import numpy as np
from PIL import Image
from functools import partial


dictionary = aruco.getPredefinedDictionary(aruco.DICT_ARUCO_ORIGINAL)

destination_dimensions = np.array([
    [0, 0],
    [4000, 0],
    [4000, 3000],
    [0, 3000],

], dtype = 'float32')

id_mapping = {
    0: 'top-left',
    1: 'top-right',
    2: 'bottom-right',
    3: 'bottom-left',
}

def correct_orientation(image):
    corners, identifiers, _ = aruco.detectMarkers(image, dictionary)
    corners_array = np.array(corners).squeeze()
    middle = corners_array.mean(axis = 1).mean(axis = 0)
    extreme_corners = np.array(list(map(partial(extreme, middle), corners_array)))
    points = dict(zip(list(identifiers.squeeze()), extreme_corners))

    corners_to_transform = np.array([
        points[0],
        points[1],
        points[2],
        points[3]
    ], dtype = 'float32')

    M = cv2.getPerspectiveTransform(corners_to_transform, destination_dimensions)
    return cv2.warpPerspective(image, M, (4000, 3000))


def fill_in_corners(points):
    for i in id_mapping.keys():
        if not i in points:
            point = input('what is the most extreme corner of the {} marker? '.format(id_mapping[i]))
            points[i] = [float(i) for i in point.split(',')]

    return points


def extreme(middle, box):
    """
    Find the corner of a box that's most extreme with respect to some
    centre point `middle`.
    """
    xs = box[:, 0]
    ys = box[:, 1]
    distances = np.sqrt((xs - middle[0]) ** 2 + (ys - middle[1]) ** 2)
    sorting = np.argsort(-distances)
    return box[sorting][0]
