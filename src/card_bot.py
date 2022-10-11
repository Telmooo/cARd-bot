import argparse
import cv2
import numpy as np

from android.android_camera import AndroidCamera
import config
from config import parse_config
from utils.draw import draw_grid
from utils.image_processing import binarize, contour_filter, enhance_image, extract_card_corners, extract_cards, extract_contours, template_matching

def run(params) -> None:

    if params["debug"]:
        config.DEBUG_MODE = True

    camera = AndroidCamera(
        mode=params["mode"], cpoint=args["cpoint"]
    )

    template = cv2.imread("./data/cards_normal/31.png")
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    template = enhance_image(template, params["config"])
    # template = template[0:190, 0:100]

    while True:
        try:
            og_frame = camera.read_frame()

            frame = cv2.cvtColor(og_frame, cv2.COLOR_BGR2GRAY)
            frame = enhance_image(frame, params["config"])

            thresh_frame = binarize(frame, params["config"])

            cv2.imshow("Binarized Image", thresh_frame)

            contours = extract_contours(thresh_frame, params["config"])

            cards = extract_cards(og_frame, contours, params["config"])

            card_corners = extract_card_corners(cards, params["config"])

            for card_corner in card_corners:
                match, val = template_matching(card_corner, template)
                if val > 0.8:
                    cv2.putText(card_corner, "VALID", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
                else:
                    cv2.putText(card_corner, f"INVALID={val}", (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)

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
    parser.add_argument(
        "--debug",
        type=bool, action="store_true", default=False,
        help="enable debug mode"
    )

    args = parser.parse_args()
    return vars(args)

if __name__ == "__main__":
    args = parse_args()

    args["cpoint"] = args["mode"][1]
    args["mode"] = args["mode"][0]

    args["config"] = parse_config(args["config"])

    run(args)