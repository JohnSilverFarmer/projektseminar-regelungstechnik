import cv2
import numpy as np


# Parameters
AR_DICT = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
W_WORLD, H_WORLD = 2 * 21.0, 29.7  # size of area marked by inner points of aruco markers in cm
H_IMG, W_IMG = int(H_WORLD * 50), int(W_WORLD * 50)
DST_POINTS = np.array([[0, H_IMG],
                       [0, 0],
                       [W_IMG, 0],
                       [W_IMG, H_IMG]], dtype="float32")


def _order_detections(ids, corners, image_midpoint):
    """
    For each detected corner the point closest to the image center is computed.
    The resulting points are ordered according to there corresponding id in
    ascending order.
    """
    ids2point = {}
    for mid, m_corners in zip(np.squeeze(ids), np.squeeze(corners)):
        ds = np.sqrt(np.sum((m_corners - image_midpoint) ** 2, axis=1))  # distance of corner points to center of image
        ids2point[mid] = m_corners[np.argmin(ds)]

    return np.array([ids2point[i] for i in range(4)])


def detect_markers_and_compute(img, debug_image=False):
    """
    Detect aruco markers in an image and computes a transform to transform the image into birds eye view.
    """
    if img.dim != 2:
        raise ValueError('Aruco detection requires gray scale images.')

    # detect aruco marker
    parameters = cv2.aruco.DetectorParameters_create()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    corners, ids, rejected_img_points = cv2.aruco.detectMarkers(img, AR_DICT, parameters=parameters)

    # compute point correspondences
    ordered_detections = _order_detections(ids, corners, np.float32([img.shape[0] / 2, img.shape[1] / 2]))
    M = cv2.getPerspectiveTransform(ordered_detections, DST_POINTS)

    # return a debug image if requested
    if debug_image:
        img_debug = cv2.cvt_color(img, cv2.GRAY2BGR)
        frame_markers = cv2.aruco.drawDetectedMarkers(img_debug, corners, ids)
        frame_markers = cv2.aruco.drawDetectedMarkers(frame_markers, rejected_img_points, borderColor=(100, 0, 240))
        return M, frame_markers
    else:
        return M
