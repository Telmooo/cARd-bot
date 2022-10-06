from turtle import position
from typing import Literal, Optional, Tuple

import numpy as np
import cv2
import requests


class AndroidCamera:
    mode: str
    ip: Optional[str]
    device_no: Optional[int]
    img_size: Optional[Tuple[int, int]]
    def __init__(
        self,
        mode: Literal["usb", "wifi"],
        url: Optional[str] = None,
        device_no: Optional[int] = None,
        img_size: Optional[Tuple[int, int]] = None
    ) -> None:
        assert (mode == "usb" and device_no is not None) or (mode == "wifi" and url is not None), "Must have URL/Device specified for the mode chosen"

        self.mode = mode
        self.url = url
        self.device_no = device_no
        self.img_size = tuple([int(size) for size in img_size])


        if self.mode == "usb":
            self.__init_device()

    def __init_device(self):
        self.device = cv2.VideoCapture(int(self.device_no))

    def __request_frame(self) -> np.ndarray:
        response = requests.get(self.url)
        if response.ok:
            frame = np.array(
                bytearray(response.content), dtype=np.uint8
            )
            frame = cv2.imdecode(frame, -1)
        
        else:
            frame = np.array([])

        return frame

    def __read_device_frame(self) -> np.ndarray:
        _ret, frame = self.device.read()
        
        return frame

    def read_frame(self) -> np.ndarray:
        match self.mode.lower():
            case "wifi":
                frame = self.__request_frame()
            case "usb":
                frame = self.__read_device_frame()

        if self.img_size:
            frame = cv2.resize(frame, self.img_size)
        
        return frame


def detect_marker(frame, arucoDict, arucoParams):
    (corners, _ids, _rejected) = cv2.aruco.detectMarkers(frame, arucoDict, parameters=arucoParams)

    if len(corners) > 0:    
        for (x,y) in corners[0][0]:
            cv2.drawMarker(frame, (int(x), int(y)), color=(0,255,0), markerType=cv2.MARKER_TILTED_CROSS, thickness=2)

def print_aruco_params(paramDict):
    '''
    https://docs.opencv.org/3.4/d1/dcd/structcv_1_1aruco_1_1DetectorParameters.html#acf6a0da0ae6bcd582c51ae14263df87a
    '''
    for param in dir(paramDict):
        if not param.startswith('__'):
            print(f'{param} = {getattr(paramDict, param)}')

def template_matching(frame, template, method=cv2.TM_CCOEFF):
    w, h = template.shape[::-1]

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    res = cv2.matchTemplate(frame_gray, template, method)
    _min_val, _max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    if method is cv2.TM_SQDIFF_NORMED or method is cv2.TM_SQDIFF:
        top_left = min_loc
    else:
        top_left = max_loc

    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv2.rectangle(frame, top_left, bottom_right, 255, 2)

    return frame

# def feature_point_detection(frame, method="sift"):
#     frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # if method == 'sift':
#     sift = cv2.SIFT_create()
#     kp_sift, des_sift = sift.detectAndCompute(frame_gray, None)  
#     # elif method == 'brief':
#     star = cv2.xfeatures2d.StarDetector_create()
#     brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
#     kp = star.detect(frame_gray,None)
#     kp_brief, des_brief = brief.compute(frame_gray, kp)

#     # print(f'Sift kp:{des_sift}') #des:{des_sift.shape} type:{kp_sift.dtype}')
#     # print(f'Brief kp:{des_brief}')# des:{des_brief.shape} type:{kp_brief.dtype}')
#     # print("=======================================")
#     # # elif method == 'orb':
#     # #     orb = cv2.ORB_create(nfeatures=2000)
#     # #     kp, des = orb.detectAndCompute(frame_gray, None)

#     # # frame = cv2.drawKeypoints(frame, kp, frame)
#     print(des_brief)
#     return kp_brief, np.float32(des_brief) if des_brief is not None else None

# def feature_point_detection(frame, template, template_kp, template_des, fp_model=None):

#     if fp_model is None:
#         fp_model = cv2.SIFT_create()
    
#     frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     frame_kp, frame_des = fp_model.detectAndCompute(frame_gray, None)

#     if frame_kp is not None and frame_des is not None:
#         FLANN_INDEX_KDTREE = 1
#         index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
#         search_params = dict(checks = 50)

#         flann = cv2.FlannBasedMatcher(index_params, search_params)
#         matches = flann.knnMatch(template_des, frame_des, k=2)

#         good = []
#         for m,n in matches:
#             if m.distance < 0.7*n.distance:
#                 good.append(m)

#         if len(good)>5:
#             src_pts = np.float32([ template_kp[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
#             dst_pts = np.float32([ frame_kp[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
#             M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
#             matchesMask = mask.ravel().tolist()
#             pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
#             dst = cv2.perspectiveTransform(pts,M)
#             frame = cv2.polylines(frame,[np.int32(dst)],True,255,3, cv2.LINE_AA)
#         else:
#             # print( "Not enough matches are found - {}/{}".format(len(good), 5) )
#             matchesMask = None
        
#         draw_params = dict(matchColor = (0,255,0), # draw matches in green color
#             singlePointColor = None,
#             matchesMask = matchesMask, # draw only inliers
#             flags = 2)
#         frame = cv2.drawMatches(template,template_kp,frame,frame_kp,good,None,**draw_params)
        
from skimage import filters
def pre_processing(frame):
    
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame, ksize=(3,3), sigmaX=20, sigmaY=00)
    clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(2,2))
    gray_frame = clahe.apply(gray_frame)

    
    # thresh_frame = cv2.adaptiveThreshold(gray_frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C , cv2.THRESH_BINARY_INV, 3, 3)
    thresh = (gray_frame > 200).astype(np.uint8)
    thresh_frame = filters.apply_hysteresis_threshold(gray_frame, 150, 200)
    thresh_frame = np.bitwise_or(thresh, thresh_frame, dtype=np.uint8)
    thresh_frame *= 255

    # edges = cv2.Canny(gray_frame, 100, 200)
    
    contours, hierarchy = cv2.findContours(thresh_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    
    approx = [cv2.approxPolyDP(curve, 0.02*cv2.arcLength(curve, True), True) for curve in contours]
    
    filtered_contours = []
    for contour in approx:
        if len(contour) == 4:
            filtered_contours.append(contour)

    # Homography
    w, h = 500, 726
    target_pts = np.float32([ [0,0],[w-1,0],[w-1,h-1],[0,h-1] ]).reshape(-1,1,2)

    card = np.zeros_like(frame)
    for contour in filtered_contours:
        anchor = contour[0][0]
        min_dist = np.inf
        max_dist = -1
        pair = -1
        for i, corner in enumerate(contour[1:], start=1):
            dist = np.sum((anchor - corner[0]) ** 2)
            if dist < min_dist:
                pair = i
                min_dist = dist

        pairs = [(0, pair)]
        pairs.append(tuple(set([0, 1, 2, 3]) - set(pairs[0])))
        src_pts = []
        for pair in pairs:
            src_pts.append([contour[pair[1]], contour[pair[0]]])
        src_pts = np.float32(src_pts).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, target_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        card = cv2.warpPerspective(frame, M, (500, 726))


        break


    return card
    


if __name__ == "__main__":

    camera = AndroidCamera("usb", device_no=1, img_size=(1280,720))

    arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_ARUCO_ORIGINAL)
    arucoParams = cv2.aruco.DetectorParameters_create()
    # sift = cv2.SIFT_create()

    # template = cv2.imread('data/simple/1c.jpg')
    # template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    # template_kp, template_des = sift.detectAndCompute(template_gray, None)  

    # h,w = template.shape[:2]

    # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    while True:
        try:
            frame = camera.read_frame()
            
            gray_frame = pre_processing(frame)
            
            # frame = feature_point_detection(frame, template, template_kp, template_des, fp_model=sift)

            # detect_marker(frame, arucoDict, arucoParams)

            # template_matching(frame, template, cv2.TM_CCOEFF)
            
            cv2.imshow("video", gray_frame)

            if (cv2.waitKey(1) == 27):
    
                break
        except InterruptedError:
            break

    cv2.destroyAllWindows()
    