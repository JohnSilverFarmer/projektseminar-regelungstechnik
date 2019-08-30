import cv2
import numpy as np


# Parameters
AR_DICT = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)


def _order_detections(ids, corners):
    """
    For each detected corner the point closest to the image center is computed.
    The resulting points are ordered according to there corresponding id in
    ascending order.
    """
    ids2point = {}
    for mid, m_corners in zip(np.squeeze(ids), np.squeeze(corners)):
        corner_idx = mid + 1
        if corner_idx > 3:
            corner_idx -= 4
        ids2point[mid] = m_corners[corner_idx]

    # check if all four markers have been detected
    num_detected = len(ids2point.keys())
    if num_detected != 4:
        raise ValueError('Four markers are required but only {} were detected. Ensure that ArUco markers are not '
                         'occluded by other objects in the scene.'.format(num_detected))

    return np.array([ids2point[i] for i in range(4)])


def detect_markers_and_compute(img, dst_size, debug_image=False):
    """
    Detect aruco markers in an image and computes a transform to transform the image into birds eye view.
    """
    if img.ndim != 2:
        raise ValueError('Aruco detection requires gray scale images.')

    # detect aruco marker
    parameters = cv2.aruco.DetectorParameters_create()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    corners, ids, rejected_img_points = cv2.aruco.detectMarkers(img, AR_DICT, parameters=parameters)

    # compute point correspondences
    ordered_detections = _order_detections(ids, corners)

    h_img, w_img = dst_size
    dst_points = np.array([[0, h_img],
                           [0, 0],
                           [w_img, 0],
                           [w_img, h_img]], dtype="float32")
    M = cv2.getPerspectiveTransform(ordered_detections, dst_points)

    # return a debug image if requested
    if debug_image:
        img_debug = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        frame_markers = cv2.aruco.drawDetectedMarkers(img_debug, corners, ids)
        frame_markers = cv2.aruco.drawDetectedMarkers(frame_markers, rejected_img_points, borderColor=(100, 0, 240))
        return M, frame_markers
    else:
        return M
