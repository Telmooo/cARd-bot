import argparse
from typing import Any, List, Tuple

import cv2
import numpy as np

from android.android_camera import AndroidCamera

def run(
    mode: str,
    cpoint: Any,
    chess_shape: Tuple[int, int],
    chess_mm: float,
    save_dir: str,
    save_frames: bool = False,
) -> None:

    camera = AndroidCamera(mode=mode, cpoint=cpoint)

    saved_frames: List[np.ndarray] = []

    print(
        "Starting calibration process...\n"
        "Recommended at least 10 captures before calibrating\n"
    )

    while True:
        frame = camera.read_frame()
        
        print(f"Captured frames [{len(saved_frames)}/10]\r")

        match cv2.waitKey(0):
            case 99:  # C
                saved_frames.append(frame.copy())

                
            case 13:  # ENTER
                print("\nFinished capturing... Beginning of calibration")
                break
            case 27:  # ESC
                print("\nAborting...")
                exit(0)


    pass

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        nargs=2, metavar=("MODE", "CPOINT"),
        help="\n".join([
            "specify type of acquisition and connection point",
            "- MODE=usb - connection point is device number",
            "- MODE=wifi - connection point is the URL to access frame"
        ])
    )
    parser.add_argument(
        "--chess-shape",
        nargs=3, metavar=("CHESS_ROWS", "CHESS_COLS", "CHESS_MM"),
        help="chessboard size, rows and columns, and size of square in milimeters. E.g. (ROWS, COLS, MM)"
    )
    parser.add_argument(
        "--save-frames",
        type=bool, action="store_true", default=False,
        help="save captured frames for calibration"
    )
    parser.add_argument(
        "--save-dir",
        type=str, default="./camera",
        help="directory to save callibration acquired and captured frames"
    )

    args = parser.parse_args()
    return vars(args)


if __name__ == "__main__":
    
    args = parse_args()

    args["cpoint"] = args["mode"][1]
    args["mode"] = args["mode"][0]
    args["chess_mm"] = args["chess_shape"][2]
    args["chess_shape"] = args["chess_shape"][:2]

