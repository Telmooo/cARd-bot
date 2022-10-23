
import os
from typing import Dict, Tuple
import cv2

import numpy as np

from sueca import Suit, Rank

def load_split_rank_suit_dataset(
    ranks_dir: str,
    suits_dir: str,
    load_colour: bool = False
) -> Tuple[Dict[Rank, np.ndarray], Dict[Suit, np.ndarray]]:

    ranks: Dict[Rank, np.ndarray] = {}
    suits: Dict[Suit, np.ndarray] = {}

    load_flag = cv2.IMREAD_COLOR if load_colour else cv2.IMREAD_GRAYSCALE

    for rank in Rank:
        rank_image = cv2.imread(os.path.join(ranks_dir, f"{rank.value}.png"), flags=load_flag)
        ranks[rank] = rank_image

    for suit in Suit:
        suit_image = cv2.imread(os.path.join(suits_dir, f"{suit.value}.png"), flags=load_flag)
        suits[suit] = suit_image

    return ranks, suits
