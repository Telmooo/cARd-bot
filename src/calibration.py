
import glob
import cv2 as cv
import numpy as np

from typing import Optional, List, Tuple

def calibrate_camera(
    images: List[np.ndarray],
    chessboard_size: Tuple[int, int],
    square_size_mm: float,
) -> Optional[Tuple]:
    assert len(img_paths) > 0, "Must have at least one image path"
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

        grayscale = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        found, corners = cv.findChessboardCorners(grayscale, chessboard_size)

        if found:
            obj_points.append(points)
            img_points.append(corners)

            # cv.drawChessboardCorners(img, chessboard_size, corners, found)
            # cv.imshow('img', img)
            # cv.waitKey(0)
    
    ret = cv.calibrateCamera(obj_points, img_points, img_size, None, None)
    return ret[1:] if ret[0] else None

if __name__ == "__main__":
    chessboard_size = (9, 6)  # (Columns, Rows)
    square_size_mm = 20

    img_paths = glob.glob("*.jpg")
    images = map(cv.imread, img_paths)
    camera_matrix, dist_coeffs, r_vecs, t_vecs = calibrate_camera(images, chessboard_size, square_size_mm)

    np.set_printoptions(suppress=True)
    print(np.array2string(camera_matrix, precision=2))
