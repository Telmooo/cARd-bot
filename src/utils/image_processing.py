from typing import Any, Dict, List, Union
from nptyping import Float32, NDArray, Shape, UInt8

import numpy as np
import cv2
from skimage import filters

from utils.geometry import Quadrants, angle_quadrant, distance_to_line, line_intersection

Grayscale8UImageType = NDArray[Shape["N, M"], UInt8]
Grayscale32FImageType = NDArray[Shape["N, M"], Float32]
Colour8UImageType = NDArray[Shape["N, M, 3"], UInt8]
Colour32FUImageType = NDArray[Shape["N, M, 3"], Float32]

GrayscaleImageType = Union[Grayscale8UImageType, Grayscale32FImageType]
ColourImageType = Union[Colour8UImageType, Colour32FUImageType]

def detect_marker(image, arucoDict, arucoParams):
    marker_image = image.copy()
    (corners, _ids, _rejected) = cv2.aruco.detectMarkers(marker_image, arucoDict, parameters=arucoParams)

    if len(corners) > 0:    
        for (x,y) in corners[0][0]:
            cv2.drawMarker(marker_image, (int(x), int(y)), color=(0,255,0), markerType=cv2.MARKER_TILTED_CROSS, thickness=2)

    return marker_image

def print_aruco_params(paramDict):
    '''
    https://docs.opencv.org/3.4/d1/dcd/structcv_1_1aruco_1_1DetectorParameters.html#acf6a0da0ae6bcd582c51ae14263df87a
    '''
    for param in dir(paramDict):
        if not param.startswith('__'):
            print(f'{param} = {getattr(paramDict, param)}')

        
def enhance_image(gray_image: Grayscale8UImageType, params: Dict[str, Any]):
    enhanced_image = cv2.GaussianBlur(
        gray_image,
        ksize=(params["gaussianBlur.kernelSize"], params["gaussianBlur.kernelSize"]),
        sigmaX=params["gaussianBlur.sigmaX"],
        sigmaY=params["gaussianBlur.sigmaY"]
    )

    clahe = cv2.createCLAHE(
        clipLimit=params["clahe.clipLimit"],
        tileGridSize=(params["clahe.tileGridSize"], params["clahe.tileGridSize"])
    )
    enhanced_image = clahe.apply(enhanced_image)

    return enhanced_image

def binarize(gray_image: Grayscale8UImageType, params: Dict[str, Any]):
    thresholded_image = filters.apply_hysteresis_threshold(
        gray_image,
        low=params["hysteresis.lowerThresh"],
        high=params["hysteresis.highThresh"]
    )

    thresholded_image = np.uint8(thresholded_image * 255)

    return thresholded_image

def extract_contours(binary_image: Grayscale8UImageType, params: Dict[str, Any]):
    contours, _hierarchy = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    contours = [
        cv2.approxPolyDP(curve, params["contours.approxPolyDPEpsilon"] * cv2.arcLength(curve, True), True)
            for curve in contours
    ]

    return contours

def detect_corners_polygonal_approximation(binary_image: Grayscale8UImageType) -> List[NDArray]:
    # Find the full contours of the binarized image
    contours, _hierarchy = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    res = []

    for contour in contours:
        # Pick two points separated by approximately half the contour length
        half = len(contour) // 2
        a, b = contour[0][0], contour[half][0]

        def farthest_along_contour(p1, p2, start=0, end=None):
            if end is None:
                end = len(contour) - 1

            max_dist = 0
            i = start

            while i != end:
                dist = distance_to_line(p1, p2, contour[i][0])
                if dist >= max_dist:
                    idx = i
                    max_dist = dist

                i = (i + 1) % len(contour)

            return idx

        # Find the points C and D that are most distant to line AB
        c_idx = farthest_along_contour(a, b, end=half - 1)
        d_idx = farthest_along_contour(a, b, half)

        c, d = contour[c_idx], contour[d_idx]

        # Find the points E and F that are most distant to line CD
        e_idx = farthest_along_contour(c[0], d[0], c_idx, d_idx - 1)
        f_idx = farthest_along_contour(c[0], d[0], d_idx, c_idx - 1)

        e, f = contour[e_idx], contour[f_idx]

        # Corners of the quadrilateral are C, D, E and F
        res.append(np.array([c, e, d, f]))

    return res

def get_quadrilateral_ord_corners(contour):
    anchor_point = contour[0][0] # Top left
    # Find closest corner
    min_distance, max_distance, closest, farthest = np.inf, 0, -1, -1
    for idx, corner in enumerate(contour[1:], start=1):
        distance = np.sum((corner[0] - anchor_point) ** 2)
        if distance < min_distance:
            min_distance = distance
            closest = idx
        if distance > max_distance:
            max_distance = distance
            farthest = idx 

    # Create the pairs
    pairs = [
        (contour[0][0], contour[closest][0]),
        (contour[farthest][0], contour[6 - closest - farthest][0])
    ]
    
    card_center = line_intersection(
        line_1=(pairs[0][0], pairs[1][0]),  # Anchor and Farthest points
        line_2=(pairs[0][1], pairs[1][1])  # Closest and 2nd Closest
    )

    # Create vectors in relation to center of card
    anchor_vector = pairs[0][0] - card_center
    closest_vector = pairs[0][1] - card_center

    # Calculate vector orientation angle [-π, π]
    anchor_angle = np.arctan2(anchor_vector[1], anchor_vector[0])
    closest_angle = np.arctan2(closest_vector[1], closest_vector[0])

    # Normalize angle
    if angle_quadrant(anchor_angle) == Quadrants.THIRD and angle_quadrant(closest_angle) == Quadrants.SECOND:
        anchor_angle += np.pi * 2
    if angle_quadrant(anchor_angle) == Quadrants.SECOND and angle_quadrant(closest_angle) == Quadrants.THIRD:
        closest_angle += np.pi * 2

    # Check if card flow orientation is in correct order, if not, reverse it
    if anchor_angle > closest_angle:  # Incorrect flow
        pairs[0] = pairs[0][::-1]
        pairs[1] = pairs[1][::-1]

    out_pts = np.float32([pairs]).reshape(-1, 1, 2)

    return out_pts


def contour_filter(contour, params, eps=1e-6):
    if len(contour) != 4:
        return False

    # corners = get_quadrilateral_ord_corners(contour)

    # n_corners = len(corners)

    # for idx, corner_1 in enumerate(corners[:2]):
    #     corner_2 = corners[(idx + 1) % n_corners]
    #     corner_3 = corners[(idx + 2) % n_corners]
    #     corner_4 = corners[(idx + 3) % n_corners]

    #     slope_12 = (corner_1[0][0] - corner_2[0][0]) / (corner_1[0][1] - corner_2[0][1] + eps)
    #     slope_43 = (corner_4[0][0] - corner_3[0][0]) / (corner_4[0][1] - corner_3[0][1] + eps)

    #     debug_img = np.zeros(shape=(480, 640, 3))

    #     cv2.circle(debug_img, (int(corner_1[0][0]), int(corner_1[0][1])), 5, (0, 0, 255), 2)
    #     cv2.circle(debug_img, (int(corner_2[0][0]), int(corner_2[0][1])), 5, (0, 0, 255), 2)

    #     cv2.line(debug_img, (int(corner_1[0][0]), int(corner_1[0][1])), (int(corner_2[0][0]), int(corner_2[0][1])), (0, 255, 255), 2)

    #     cv2.circle(debug_img, (int(corner_3[0][0]), int(corner_3[0][1])), 5, (255, 0, 0), 2)
    #     cv2.circle(debug_img, (int(corner_4[0][0]), int(corner_4[0][1])), 5, (255, 0, 0), 2)
    #     cv2.line(debug_img, (int(corner_3[0][0]), int(corner_3[0][1])), (int(corner_4[0][0]), int(corner_4[0][1])), (255, 255, 0), 2)


    #     slope_ratio = slope_12 / (slope_43 + eps)

    #     error = slope_ratio - 1.0

    #     cv2.putText(debug_img, f"SLOPE={slope_12:.5f}", np.int32((corner_1[0] + corner_2[0]) / 2 - np.array([50, 100])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    #     cv2.putText(debug_img, f"SLOPE={slope_43:.5f}", np.int32((corner_3[0] + corner_4[0]) / 2 - np.array([50, 100])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)

    #     cv2.putText(debug_img, f"ERROR={error:.5f}", (300, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    #     cv2.imshow("Debug slope test", debug_img)

    #     if abs(error) > params["filter.slopeThresh"]:
    #         return False

    #     break

    return True

def extract_cards(image, contours, params):
    
    # Filter contours
    filtered_contours = list(filter(lambda x: contour_filter(x, params), contours))

    target_pts = np.float32([
        [0, 0], # Top left
        [params["cards.width"] - 1, 0], # Top right
        [params["cards.width"] - 1, params["cards.height"] - 1], # Bottom right
        [0, params["cards.height"] - 1], # Bottom left
    ]).reshape(-1, 1, 2)

    cards = []
    debug_img = np.zeros_like(image)
    COLOURS = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]
    for contour in filtered_contours:
        src_pts = get_quadrilateral_ord_corners(contour)

        for idx, corner in enumerate(src_pts):
            cv2.circle(debug_img, np.int32(corner[0]), 5, COLOURS[idx], 2)


        # M = cv2.perspectiveTransform(src_pts, target_pts)
        M, _mask = cv2.findHomography(src_pts, target_pts, cv2.RANSAC, 5.0)
        card = cv2.warpPerspective(image, M, (params["cards.width"], params["cards.height"]))
        
        cards.append(card)

    cv2.imshow("Corner Order", debug_img)
    return cards

def extract_card_corners(cards, params):
    card_corners = [
        card[
            0:params["cards.cornerHeight"],
            0:params["cards.cornerWidth"]
        ] for card in cards
    ]

    return card_corners

def extract_card_rank_suit(cards, params):
    cards_rank_suit = [
        (
            card[
                0:params["cards.splitHeight"]+5,
                0:params["cards.cornerWidth"]+5
            ],
            card[
                params["cards.splitHeight"]-5:params["cards.cornerHeight"]+5,
                0:params["cards.cornerWidth"]+5
            ]
        ) for card in cards
    ]

    return cards_rank_suit

def template_matching(
    frame: Union[GrayscaleImageType, ColourImageType],
    template: Union[GrayscaleImageType, ColourImageType],
    method: int = cv2.TM_CCOEFF_NORMED, temp="ls"
):
    w, h = template.shape[::-1]

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    thresh_frame = np.uint8(frame_gray < 127) * 255
    thresh_template = np.uint8(template < 127) * 255
    res = cv2.matchTemplate(thresh_frame, thresh_template, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    if method is cv2.TM_SQDIFF_NORMED or method is cv2.TM_SQDIFF:
        top_left = min_loc
        match_val = min_val
    else:
        top_left = max_loc
        match_val = max_val

    match = frame[
        top_left[1]:top_left[1] + w,
        top_left[0]:top_left[0] + h
    ]

    cv2.imshow(f"{temp}FRAME_DEBUG", thresh_frame)
    cv2.imshow(f"{temp}TEMPLATE_DEBUG", thresh_template)
    cv2.moveWindow(f"{temp}TEMPLATE_DEBUG", 500, 500 + 500 * (temp != "rank"))
    cv2.moveWindow(f"{temp}FRAME_DEBUG", 1000, 500 + 500 * (temp != "rank"))

    return match, match_val