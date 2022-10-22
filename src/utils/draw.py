
from typing import List, Optional, Tuple, Union
import numpy as np
import cv2

from data.load_dataset import Suit
from sueca import SuecaGame

def draw_grid(images: Union[np.ndarray, List[np.ndarray]], resize: Optional[Tuple[int, int]] = None):
    if not images:
        return np.array([])
    
    grid = np.array(images)
    
    if grid.ndim == 3:
        n_images, height, width = grid.shape
        channels = 1
        grid = grid.reshape(n_images, height, width, -1)
    else:
        n_images, height, width, channels = grid.shape

    cols = int(np.ceil(np.sqrt(n_images)))
    rows = int(np.ceil(n_images / cols))

    if n_images < cols * rows:
        grid = np.concatenate((
            grid,
            np.zeros(
                shape=(rows * cols - n_images, height, width, channels),
                dtype=grid.dtype
            )
        ), axis=0)

    grid = grid.reshape(rows, cols, height, width, channels) \
                .swapaxes(1, 2)\
                .reshape(height * rows, width * cols, channels)

    grid_dims = grid.shape[:2]
    if resize is not None and grid_dims > resize:
        grid = cv2.resize(grid, resize)
    
    return grid

def draw_scores(dst_image, sueca_game: SuecaGame, round_suit: Suit, pos: Tuple[int, int]):
    THICKNESS = 2
    COLOR = (85, 135, 0)
    SCALE = 0.6

    # Team 1 score
    cv2.putText(
        img=dst_image,
        text=f"Round #{sueca_game.rounds_evaluated + 1}",
        org=pos,
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=SCALE,
        color=COLOR,
        thickness=THICKNESS,
        lineType=cv2.LINE_AA
    )

    # Team 1 score
    cv2.putText(
        img=dst_image,
        text=f"TEAM 1: {sueca_game.team_points[0]}",
        org=(pos[0], pos[1] + 50),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=SCALE,
        color=COLOR,
        thickness=THICKNESS,
        lineType=cv2.LINE_AA
    )

    # Team 2 score
    cv2.putText(
        img=dst_image,
        text=f"TEAM 2: {sueca_game.team_points[1]}",
        org=(pos[0], pos[1] + 100),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=SCALE,
        color=COLOR,
        thickness=THICKNESS,
        lineType=cv2.LINE_AA
    )

def draw_winner(dst_image, sueca_game : SuecaGame, card_center_labels, pos : Tuple[int, int]):
    contours = [x[3] for x in card_center_labels]
    contours = [c for i, c in enumerate(contours) if i % 2 == sueca_game.winner()]

    text = "TIE"
    COLOR = (0, 0, 255)
    if sueca_game.winner() is not None:
        text = f"TEAM {sueca_game.winner()+1} WINS"
        COLOR = (85, 135, 0)

    cv2.putText(
        img=dst_image, 
        text=text,
        org=pos,
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.8,
        color=COLOR,
        thickness=4,
        lineType=cv2.LINE_AA
    )

    cv2.drawContours(dst_image, contours, -1, (0, 255, 255), 2)