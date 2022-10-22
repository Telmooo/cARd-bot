import argparse
import os
from typing import Dict, Tuple
import cv2
import numpy as np

from android.android_camera import AndroidCamera
import config
from config import parse_config
from data.load_dataset import Rank, Suit, load_split_rank_suit_dataset
from opengl.render import AR_Render
from sueca import Card, SuecaGame, SuecaRound
from utils.draw import draw_grid
from utils.image_processing import *

def run(params) -> None:

    if params["debug"]:
        config.DEBUG_MODE = True

    camera = AndroidCamera(
        mode=params["mode"], cpoint=args["cpoint"]
    )

    ar_renderer = AR_Render(camera, './src/opengl/models/LPC/Low_Poly_Cup.obj', 0.05)
    # ar_renderer = AR_Render(camera, './src/opengl/models/plastic_cup/Plastic_Cup.obj', 0.02)


    dataset_ranks, dataset_suits = load_split_rank_suit_dataset(
        ranks_dir=os.path.join(params["config"]["cards.dataset"], "./ranks"),
        suits_dir=os.path.join(params["config"]["cards.dataset"], "./suits"),
        load_colour=False
    )


    sueca_game = SuecaGame(Suit.Clubs)

    # cv2.imshow("Rank Templates", draw_grid(list(dataset_ranks.values())))
    # print("Rank Templates", [key.name for key in dataset_ranks.keys()])
    # cv2.imshow("Suit Templates", draw_grid(list(dataset_suits.values())))
    # print("Suit Templates", [key.name for key in dataset_suits.keys()])

    # for rank in dataset_ranks.keys():
    #     dataset_ranks[rank] = enhance_image(dataset_ranks[rank], params["config"])

    # for suit in dataset_suits.keys():
    #     dataset_suits[suit] = enhance_image(dataset_suits[suit], params["config"])

    while True:
        try:
            orig_frame = camera.read_frame()
            frame = cv2.cvtColor(orig_frame, cv2.COLOR_BGR2GRAY)
            frame = enhance_image(frame, params["config"])

            thresh_frame = binarize(frame, params["config"])

            # if config.DEBUG_MODE:
            #     cv2.imshow("Binarized Image", thresh_frame)

            # contours = extract_contours(thresh_frame, params["config"])
            contours = detect_corners_polygonal_approximation(thresh_frame)

            cards, card_centers = extract_cards(orig_frame, contours, params["config"])

            # if cards:
            #     cv2.imshow("Cards", draw_grid(cards))

            cards_rank_suit = extract_card_rank_suit(cards, params["config"])

            cards_labels = []
            cards_center_label = []
            for idx, (card_rank, card_suit) in enumerate(cards_rank_suit):
                red = is_red_suit(card_suit)

                # TODO: refactor this
                def binarize_rank_suit(img):
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    _ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    return img

                card_rank = binarize_rank_suit(card_rank)
                card_suit = binarize_rank_suit(card_suit)

                top, left = int(0.05 * card_rank.shape[0]), int(0.05 * card_rank.shape[1])
                card_rank = cv2.copyMakeBorder(card_rank, top, top, left, left,
                        cv2.BORDER_CONSTANT, None, (255,255,255))

                top, left = int(0.05 * card_suit.shape[0]), int(0.05 * card_suit.shape[1])
                card_suit = cv2.copyMakeBorder(card_suit, top, top, left, left,
                        cv2.BORDER_CONSTANT, None, (255,255,255))

                # cv2.imshow(f"Rank - {idx}", card_rank)
                # cv2.imshow(f"Suit - {idx}", card_suit)
                # cv2.moveWindow(f"Rank - {idx}", 100*idx, 0)
                # cv2.moveWindow(f"Suit - {idx}", 100*idx, 100)

                rank_match: Dict[Rank, Tuple[float, np.ndarray]] = {}
                for rank, template_rank in dataset_ranks.items():
                    match, match_val = template_matching(card_rank, template_rank, method=cv2.TM_CCORR_NORMED, temp="rank")
                    rank_match[rank] = (match_val, match)

                suit_match: Dict[Suit, Tuple[float, np.ndarray]] = {}
                for suit, template_suit in dataset_suits.items():
                    # TODO: refactor
                    if (red and suit in {Suit.Clubs, Suit.Spades}) or (not red and suit in {Suit.Hearts, Suit.Diamonds}):
                        continue

                    match, match_val = template_matching(card_suit, template_suit, method=cv2.TM_CCORR_NORMED, temp="suit")
                    suit_match[suit] = (match_val, match)

                max_rank_match = max(rank_match, key=lambda k: rank_match[k][0])
                max_suit_match = max(suit_match, key=lambda k: suit_match[k][0])

                cards_labels.append(
                    ((max_rank_match.name,) + rank_match[max_rank_match],
                     (max_suit_match.name,) + suit_match[max_suit_match]
                    )
                )

                cards_center_label.append((card_centers[idx], max_rank_match, max_suit_match))

            filtered_contours = list(filter(lambda x: contour_filter(x, params["config"]), contours))

            cards_center_label = sorted(cards_center_label, key=lambda x : x[0][0]+x[0][1]) # sort clock-wise (l - t - r - b)

            if config.DEBUG_MODE:
                debug_frame = orig_frame.copy()
                cv2.drawContours(debug_frame, filtered_contours, -1, (0, 0, 255), 2)

                for idx, label in enumerate(cards_labels):
                    cv2.putText(
                        img=debug_frame,
                        text=f"{label[0][0]} of {label[1][0]}",
                        org=(filtered_contours[idx][0][0] - np.array([-50, 50])),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        color=(85, 135, 0),
                        thickness=2,
                        lineType=cv2.LINE_AA
                    )
                    cv2.circle(debug_frame, np.int32(card_centers[idx]), 3, (255, 255, 0), 1)
                    # cv2.putText(
                    #     img=debug_frame,
                    #     text=f"CONFIDENCE={label[0][1]:.3f} | {label[1][1]:.3f}",
                    #     org=(filtered_contours[idx][0][0] - np.array([-50, 0])),
                    #     fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    #     fontScale=0.5,
                    #     color=(85, 135, 0),
                    #     thickness=2,
                    #     lineType=cv2.LINE_AA
                    # )

                cv2.imshow("Contours Frame", debug_frame)

                ar_renderer.set_frame(orig_frame)

                if (ar_renderer.is_round_over):
                    cards = [Card(rank, suit) for (_, rank, suit) in cards_center_label]
                    sueca_round = SuecaRound(Suit.Hearts, cards) # pick suit based on first card

                    sueca_game.evaluate_round(sueca_round)

                    if sueca_game.is_finished():
                        ar_renderer.display_obj = True
                    
                    ar_renderer.is_round_over = False

            # cv2.imshow("Camera Frame", orig_frame)

            if cards:
                pass
                # cv2.imshow("Cards", draw_grid(cards, resize=(1280, 720)))
                # cv2.imshow("Card corners", draw_grid(card_corners, resize=(960, 540)))

            if cv2.waitKey(1) == 27:
                break
        except InterruptedError:
            break

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        required=True, nargs=2, metavar=("MODE", "CPOINT"),
        help="\n".join([
            "specify type of acquisition and connection point",
            "- MODE=usb - connection point is device number",
            "- MODE=wifi - connection point is the URL to access frame"
        ]),
    )
    parser.add_argument(
        "--config",
        type=str, help="path to config.yaml",
        default="./config.yaml",
    )
    parser.add_argument(
        "--debug",
        action="store_true", default=False,
        help="enable debug mode",
    )

    args = parser.parse_args()
    return vars(args)

if __name__ == "__main__":
    args = parse_args()

    args["cpoint"] = args["mode"][1]
    args["mode"] = args["mode"][0]

    args["config"] = parse_config(args["config"])

    run(args)