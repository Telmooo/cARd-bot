
from android.android_camera import AndroidCamera
from typing import Any, List, Optional, Tuple

import argparse
import cv2
import numpy as np
import os


def calibrate_camera(
    images: List[np.ndarray],
    chessboard_size: Tuple[int, int],
    square_size_mm: float,
    show_results: bool = False,
) -> Optional[Tuple]:
    assert len(images) > 0, "Must have at least one calibration image"
    assert chessboard_size[0] > 0 and chessboard_size[1] > 0, "Chessboard sizes must be positive"
    assert square_size_mm > 0, "Square size must be positive"

    points = []
    for y in range(chessboard_size[1]):
        for x in range(chessboard_size[0]):
            points.append([x, y, 0])
    points = square_size_mm * np.array(points, np.float32)

    img_size = None
    obj_points = []
    img_points = []

    for img in images:
        if img_size is None:
            img_size = img.shape[:2][::-1]

        grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        found, corners = cv2.findChessboardCorners(grayscale, chessboard_size)

        if found:
            obj_points.append(points)
            img_points.append(corners)

            if show_results:
                cv2.drawChessboardCorners(img, chessboard_size, corners, found)
                cv2.imshow('Calibration Result', img)
                cv2.waitKey(500)

    ret = cv2.calibrateCamera(obj_points, img_points, img_size, None, None)
    return ret[1:] if ret[0] else None


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
        cv2.imshow('Camera', frame)

        match cv2.pollKey():
            case 99: # C
                saved_frames.append(frame.copy())
                print(f"Captured frames [{len(saved_frames)} / 10]\r")
            case 13:  # ENTER
                print("\nFinished capturing. Beginning calibration...")
                break
            case 27:  # ESC
                print("\nAborting...")
                exit(0)

    result = calibrate_camera(saved_frames, chess_shape, chess_mm)
    

    if result is None:
        print("Calibration failed!")
    else:
        print("Calibration was successful! Saving results...")

        os.makedirs(save_dir, exist_ok=True)

        if save_frames:
            for i, frame in enumerate(saved_frames):
                cv2.imwrite(os.path.join(save_dir, f"frame{i}.png"), frame,)

        camera_matrix = result[0]
        camera_dist_coeffs = result[1]
    
        camera_matrix.dump(os.path.join(save_dir, "camera_matrix.numpy"))
        camera_dist_coeffs.dump(os.path.join(save_dir, "camera_dist_coeffs.numpy"))


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        required=True, nargs=2, metavar=("MODE", "CPOINT"),
        help="\n".join([
            "specify type of acquisition and connection point",
            "- MODE=usb - connection point is device number",
            "- MODE=wifi - connection point is the URL to access frame"
        ])
    )
    parser.add_argument(
        "--chess-shape",
        nargs=3, metavar=("ROWS", "COLS", "MM"), default=(6, 9, 20),
        help="chessboard size, rows and columns, and size of square in milimeters. E.g. (ROWS, COLS, MM)"
    )
    parser.add_argument(
        "--save-frames",
        action="store_true", default=False,
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
    args["chess_shape"] = args["chess_shape"][:2][::-1]

    run(args["mode"], args["cpoint"], args["chess_shape"], args["chess_mm"],
            args["save_dir"], args["save_frames"])
