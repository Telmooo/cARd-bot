import argparse
import os
from typing import Dict, Tuple
import cv2
import numpy as np

from android.android_camera import AndroidCamera
import config
from config import parse_config
from data.load_dataset import Rank, Suit, load_split_rank_suit_dataset
from utils.draw import draw_grid
from utils.image_processing import binarize, contour_filter, enhance_image, extract_card_corners, extract_card_rank_suit, extract_cards, extract_contours, feature_point_matching, template_matching

def run(params) -> None:

    if params["debug"]:
        config.DEBUG_MODE = True

    camera = AndroidCamera(
        mode=params["mode"], cpoint=args["cpoint"]
    )

    dataset_ranks, dataset_suits = load_split_rank_suit_dataset(
        ranks_dir=os.path.join(params["config"]["cards.dataset"], "./ranks"),
        suits_dir=os.path.join(params["config"]["cards.dataset"], "./suits"),
        load_colour=False
    )

    cv2.imshow("Rank Templates", draw_grid(list(dataset_ranks.values())))
    print("Rank Templates", [key.name for key in dataset_ranks.keys()])
    cv2.imshow("Suit Templates", draw_grid(list(dataset_suits.values())))
    print("Suit Templates", [key.name for key in dataset_suits.keys()])
    
    # for rank in dataset_ranks.keys():
    #     dataset_ranks[rank] = enhance_image(dataset_ranks[rank], params["config"])
    
    # for suit in dataset_suits.keys():
    #     dataset_suits[suit] = enhance_image(dataset_suits[suit], params["config"])

    while True:
        try:

            og_frame = camera.read_frame()
            frame = cv2.cvtColor(og_frame, cv2.COLOR_BGR2GRAY)
            frame = enhance_image(frame, params["config"])

            thresh_frame = binarize(frame, params["config"])

            # if config.DEBUG_MODE:
            #     cv2.imshow("Binarized Image", thresh_frame)

            contours = extract_contours(thresh_frame, params["config"])

            cards = extract_cards(cv2.cvtColor(og_frame, cv2.COLOR_BGR2GRAY), contours, params["config"])


            # for card in cards:
            #     feature_point_matching(card, frame)

            cards_rank_suit = extract_card_rank_suit(cards, params["config"])


            # cards_labels = []
            # for card_rank, card_suit in cards_rank_suit:
            #     rank_match = {}
            #     for rank, template_rank in dataset_ranks.items():

            #         rank_matches, matches = feature_point_matching(template_rank, card_rank)
            #         # match, match_val = template_matching(card_rank, template_rank, method=cv2.TM_CCORR_NORMED, temp="rank")

            #         # if rank_matches is not None:
            #         #     cv2.imshow(f"Rank Matches", rank_matches)
            #         #     cv2.moveWindow(f"Rank Matches", 300, 300 + 300)
                    

            #         rank_match[rank] = (len(matches), rank_matches)
                
            #     suit_match = {}
            #     for suit, template_suit in dataset_suits.items():

            #         suit_matches, matches = feature_point_matching(template_suit, card_suit)
            #         # match, match_val = template_matching(card_suit, template_suit, method=cv2.TM_CCORR_NORMED, temp="suit")

            #         # if suit_matches is not None:
            #         #     cv2.imshow(f"Suit Matches", suit_matches)
            #         #     cv2.moveWindow(f"Suit Matches", 600, 300 + 300)
                    
            #         suit_match[suit] = (len(matches), suit_matches)


            #     max_rank_match = max(rank_match, key=lambda k: rank_match[k][0])
            #     max_suit_match = max(suit_match, key=lambda k: suit_match[k][0])

            #     cards_labels.append(
            #         (((max_rank_match.name,) +  rank_match[max_rank_match]),
            #          ((max_suit_match.name,) +  suit_match[max_suit_match])
            #         )
            #     )

            # if config.DEBUG_MODE:
            #     debug_frame = og_frame.copy()
            #     filtered_contours = list(filter(lambda x: contour_filter(x, params["config"]), contours))
            #     cv2.drawContours(debug_frame, filtered_contours, -1 ,(0, 0, 255), 2)

            #     for idx, label in enumerate(cards_labels):
            #         cv2.putText(
            #             img=debug_frame,
            #             text=f"{label[0][0]} of {label[1][0]}",
            #             org=(filtered_contours[idx][0][0] - np.array([-50, 50])),
            #             fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            #             fontScale=0.5,
            #             color=(255, 255, 255),
            #             thickness=2,
            #             lineType=cv2.LINE_AA
            #         )
            #         cv2.putText(
            #             img=debug_frame,
            #             text=f"CONFIDENCE={label[0][1]:.3f} | {label[1][1]:.3f}",
            #             org=(filtered_contours[idx][0][0] - np.array([-50, 0])),
            #             fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            #             fontScale=0.5,
            #             color=(255, 255, 255),
            #             thickness=2,
            #             lineType=cv2.LINE_AA
            #         )

            #         if (label[0][2] is not None):
            #             cv2.imshow(f"Rank Matches", label[0][2])
            #             cv2.moveWindow(f"Rank Matches", 300, 300 + 300)

            #         if label[1][2] is not None:
            #             cv2.imshow("Suit Matches", label[1][2])
            #             cv2.moveWindow("Suit Matches", 600, 300 + 300)

            #     cv2.imshow("Contours Frame", debug_frame)



            cards_labels = []
            for card_rank, card_suit in cards_rank_suit:
                rank_match: Dict[Rank, Tuple[float, np.ndarray]] = {}
                for rank, template_rank in dataset_ranks.items():
                    match, match_val = template_matching(card_rank, template_rank, method=cv2.TM_CCORR_NORMED, temp="rank")

                    rank_match[rank] = (match_val, match)
                
                suit_match: Dict[Suit, Tuple[float, np.ndarray]] = {}
                for suit, template_suit in dataset_suits.items():
                    match, match_val = template_matching(card_suit, template_suit, method=cv2.TM_CCORR_NORMED, temp="suit")
                    
                    suit_match[suit] = (match_val, match)


                max_rank_match = max(rank_match, key=lambda k: rank_match[k][0])
                max_suit_match = max(suit_match, key=lambda k: suit_match[k][0])

                cards_labels.append(
                    ((max_rank_match.name,) + rank_match[max_rank_match],
                     (max_suit_match.name,) + suit_match[max_suit_match]
                    )
                )

            if config.DEBUG_MODE:
                debug_frame = og_frame.copy()
                filtered_contours = list(filter(lambda x: contour_filter(x, params["config"]), contours))
                cv2.drawContours(debug_frame, filtered_contours, -1 ,(0, 0, 255), 2)

                for idx, label in enumerate(cards_labels):
                    cv2.putText(
                        img=debug_frame,
                        text=f"{label[0][0]} of {label[1][0]}",
                        org=(filtered_contours[idx][0][0] - np.array([-50, 50])),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        color=(255, 255, 255),
                        thickness=2,
                        lineType=cv2.LINE_AA
                    )
                    cv2.putText(
                        img=debug_frame,
                        text=f"CONFIDENCE={label[0][1]:.3f} | {label[1][1]:.3f}",
                        org=(filtered_contours[idx][0][0] - np.array([-50, 0])),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        color=(255, 255, 255),
                        thickness=2,
                        lineType=cv2.LINE_AA
                    )


                cv2.imshow("Contours Frame", debug_frame)

            
            cv2.imshow("Camera Frame", og_frame)

            if cards:
                pass
                # cv2.imshow("Cards", draw_grid(cards, resize=(1280, 720)))

                # cv2.imshow("Card corners", draw_grid(card_corners, resize=(960, 540)))

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