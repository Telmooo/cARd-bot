from typing import Any, Dict, Union
from nptyping import Float32, NDArray, Shape, UInt8

import numpy as np
import cv2
from skimage import filters

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
    pairs = [(0, closest), (farthest, 6 - closest - farthest)]
    
    out_pts = [
        [contour[pair[1]], contour[pair[0]]]
            for pair in pairs
    ]

    slope_12 = calc_slope(out_pts[0][0], out_pts[0][1])
    slope_43 = calc_slope(out_pts[1][1], out_pts[1][0])



    # out_pts = np.float32([
    #     [contour[pair[1]], contour[pair[0]]]
    #         for pair in pairs
    # ]).reshape(-1, 1, 2)

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
        [params["card.width"] - 1, 0], # Top right
        [params["card.width"] - 1, params["card.height"] - 1], # Bottom right
        [0, params["card.height"] - 1], # Bottom left
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
        card = cv2.warpPerspective(image, M, (params["card.width"], params["card.height"]))
        
        cards.append(card)

    cv2.imshow("Corner Order", debug_img)
    return cards

def extract_card_corners(cards, params):
    card_corners = [
        card[
            0:params["card.cornerHeight"],
            0:params["card.cornerWidth"]
        ] for card in cards
    ]

    return card_corners

def template_matching(
    frame: Union[GrayscaleImageType, ColourImageType],
    template: Union[GrayscaleImageType, ColourImageType],
    method: int = cv2.TM_CCOEFF_NORMED
):
    w, h = template.shape[::-1]

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    res = cv2.matchTemplate(frame_gray, template, method)
    _min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    if method is cv2.TM_SQDIFF_NORMED or method is cv2.TM_SQDIFF:
        top_left = min_loc
    else:
        top_left = max_loc

    match = frame[
        top_left[1]:top_left[1] + w,
        top_left[0]:top_left[0] + h
    ]

    return frame, max_val