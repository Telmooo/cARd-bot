
from typing import Dict

import os
import argparse
import re
from glob import glob
import cv2

RANKS_DIR = "./ranks"
SUITS_DIR = "./suits"
CARD_FILE_REGEX = "^(?P<rank>\w+)_(?P<suit>\w+).\w+$"
SUPPORTED_FORMATS = (".png", ".jpg", ".jpeg")

def setup_dataset(
    dataset_dir: str,
    output_dir: str,
    regions: Dict[str, int],
    split_rank_suit: bool = False,
) -> None:

    os.makedirs(output_dir, exist_ok=True)
    if split_rank_suit:
        ranks_dir = os.path.join(output_dir, RANKS_DIR)
        suits_dir = os.path.join(output_dir, SUITS_DIR)
        os.makedirs(ranks_dir, exist_ok=True)
        os.makedirs(suits_dir, exist_ok=True)

    files = []
    for format in SUPPORTED_FORMATS:
        files.extend(glob(os.path.join(dataset_dir, f"*{format}")))


    regex_matcher = re.compile(CARD_FILE_REGEX)

    for file in files:
        file_basename = os.path.basename(file)
        matches = regex_matcher.match(file_basename)
        if matches is None:
            continue

        image = cv2.imread(file)
        if not split_rank_suit:

            region = image[
                regions["ybase"]:regions["ylimit"],
                regions["xbase"]:regions["xlimit"],
            ]

            cv2.imwrite(os.path.join(output_dir, file_basename), region)
        else:
            extension = os.path.splitext(file_basename)[1]
            rank = matches.group("rank")
            rank_file = os.path.join(ranks_dir, f"{rank}{extension}")
            suit = matches.group("suit")
            suit_file = os.path.join(suits_dir, f"{suit}{extension}")

            rank_region = image[
                regions["yrankbase"]:regions["yranklimit"],
                regions["xrankbase"]:regions["xranklimit"],
            ]

            suit_region = image[
                regions["ysuitbase"]:regions["ysuitlimit"],
                regions["xsuitbase"]:regions["xsuitlimit"],
            ]
            if not os.path.exists(rank_file):
                cv2.imwrite(os.path.join(ranks_dir, f"{rank}{extension}"), rank_region)
            if not os.path.exists(suit_file):
                cv2.imwrite(os.path.join(suits_dir, f"{suit}{extension}"), suit_region)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        type=str, required=True,
        help="path to raw data containing cards images"
    )
    parser.add_argument(
        "--outdir",
        type=str, required=True,
        help="path to destination directory of the dataset"
    )
    parser.add_argument(
        "--split-rank-suit",
        action="store_true", default=False,
        help="split ranks and suits to different images"
    )
    parser.add_argument(
        "--regions",
        nargs="+", required=True,
        help="region specification for the area to cut"
    )

    args = parser.parse_args()
    return vars(args)
    
if __name__ == "__main__":
    args = parse_args()

    n_regions = len(args["regions"])
    dataset_dir = args["data"]
    output_dir = args["outdir"]
    split_rank_suit = args["split_rank_suit"]

    del args["data"], args["outdir"], args["split_rank_suit"]

    if not split_rank_suit:
        match n_regions:
            case 2:
                args["xbase"], args["ybase"] = 0, 0
                args["xlimit"], args["ylimit"] = args["regions"]
                del args["regions"]
            case 4:
                args["xbase"], args["ybase"], args["xlimit"], args["ylimit"] = args["regions"]
                del args["regions"]
            case other:
                exit(1)

    else:
        match n_regions:
            case 3:
                args["xrankbase"], args["yrankbase"], args["xsuitbase"] = 0, 0, 0
                args["xranklimit"], args["ysuitlimit"], args["yranklimit"] = args["regions"]
                args["xsuitlimit"] = args["xranklimit"]
                args["ysuitbase"] = args["yranklimit"]
                del args["regions"]
            case 5:
                args["xrankbase"], args["yrankbase"], args["xranklimit"], args["ysuitlimit"], args["yranklimit"] = args["regions"]
                args["xsuitbase"] = args["xrankbase"]
                args["xsuitlimit"] = args["xranklimit"]
                args["ysuitbase"] = args["yranklimit"]
            case 8:
                args["xrankbase"], args["yrankbase"], args["xranklimit"], args["yranklimit"], \
                    args["xsuitbase"], args["ysuitbase"], args["xsuitlimit"], args["ysuitlimit"] = args["regions"]
            case other:
                exit(1)

    for key in args.keys():
        args[key] = int(args[key])

    setup_dataset(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        regions=args,
        split_rank_suit=split_rank_suit
    )