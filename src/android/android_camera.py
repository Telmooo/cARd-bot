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
