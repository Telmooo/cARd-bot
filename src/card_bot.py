import argparse
import cv2

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

    while True:
        try:
            og_frame = camera.read_frame()
            frame = cv2.cvtColor(og_frame, cv2.COLOR_BGR2GRAY)
            frame = enhance_image(frame, params["config"])

            thresh_frame = binarize(frame, params["config"])

            if config.DEBUG_MODE:
                cv2.imshow("Binarized Image", thresh_frame)

            contours = extract_contours(thresh_frame, params["config"])

            cards = extract_cards(og_frame, contours, params["config"])

            card_corners = extract_card_corners(cards, params["config"])

            if config.DEBUG_MODE:
                debug_frame = og_frame.copy()
                filtered_contours = list(filter(lambda x: contour_filter(x, params["config"]), contours))
                cv2.drawContours(debug_frame, filtered_contours, -1 ,(0, 0, 255), 2)

                cv2.imshow("Contours Frame", debug_frame)
            
            cv2.imshow("Camera Frame", og_frame)

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
        action="store_true", default=False,
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