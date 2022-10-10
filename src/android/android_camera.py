
from typing import Any, Literal, Optional, Tuple

import numpy as np
import cv2
import requests


class AndroidCamera:
    mode: str
    cpoint: Any
    img_size: Optional[Tuple[int, int]]
    def __init__(
        self,
        mode: Literal["usb", "wifi"],
        cpoint: Any,
        img_size: Optional[Tuple[int, int]] = None
    ) -> None:
        self.mode = mode

        match self.mode:
            case "usb":
                self.cpoint = int(cpoint)
            case "wifi":
                self.cpoint = str(cpoint)
                
        self.img_size = tuple([int(size) for size in img_size]) if img_size else None
        if self.mode == "usb":
            self.__init_device()

    def __init_device(self):
        self.device = cv2.VideoCapture(int(self.cpoint))

    def __request_frame(self) -> np.ndarray:
        response = requests.get(self.cpoint)
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


if __name__ == "__main__":

    camera = AndroidCamera("usb", device_no=1, img_size=(1280,720))

    arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_ARUCO_ORIGINAL)
    arucoParams = cv2.aruco.DetectorParameters_create()

    print_aruco_params(arucoParams)
    while True:
        try:
            frame = camera.read_frame()

            detect_marker(frame, arucoDict, arucoParams)
            
            cv2.imshow("video", frame)

            if (cv2.waitKey(1) == 27):
    
                break
        except InterruptedError:
            break

    cv2.destroyAllWindows()
    