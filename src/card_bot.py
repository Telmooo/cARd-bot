import argparse
import cv2
import numpy as np

from android.android_camera import AndroidCamera
from config import parse_config
from utils.draw import draw_grid
from utils.image_processing import binarize, contour_filter, enhance_image, extract_card_corners, extract_cards, extract_contours

def run(params) -> None:

    camera = AndroidCamera(
        mode=params["mode"], cpoint=args["cpoint"]
    )

    while True:
        try:
            og_frame = camera.read_frame()

            frame = cv2.cvtColor(og_frame, cv2.COLOR_BGR2GRAY)
            frame = enhance_image(frame, params["config"])

            thresh_frame = binarize(frame, params["config"])
            contours = extract_contours(thresh_frame, params["config"])

            cards = extract_cards(og_frame, contours, params["config"])

            card_corners = extract_card_corners(cards, params["config"])

            filtered_contours = list(filter(lambda x: contour_filter(x, params["config"]), contours))
            cv2.drawContours(og_frame, filtered_contours, -1 ,(0, 0, 255), 2)

            cv2.imshow("Frame", og_frame)

            if cards:
                cv2.imshow("Cards", draw_grid(cards, resize=(1280, 720)))

                cv2.imshow("Card corners", draw_grid(card_corners, resize=(960, 540)))

            if cv2.waitKey(1) == 27:
                break

        except InterruptedError:
            break

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
        ]),
        required=True
    )
    parser.add_argument(
        "--config",
        type=str, help="path to config.yaml",
        default="./config.yaml",
        required=True
    )

    args = parser.parse_args()
    return vars(args)

if __name__ == "__main__":
    args = parse_args()

    args["cpoint"] = args["mode"][1]
    args["mode"] = args["mode"][0]

    args["config"] = parse_config(args["config"])

    run(args)