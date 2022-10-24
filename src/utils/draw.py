
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

def draw_scores(img, pos: Tuple[int, int], sueca_game: SuecaGame, round_suit: Suit, error_str: str):
    def draw_text(text: str, pos: Tuple[int, int], color = (85, 135, 0)):
        cv2.putText(
            img=img,
            text=text,
            org=pos,
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.6,
            color=color,
            thickness=1,
            lineType=cv2.LINE_AA
        )

    # Trump suit
    draw_text(f"Trump Suit: {sueca_game.trump_suit.name}", pos)

    # Round number
    draw_text(f"Round #{sueca_game.rounds_evaluated + 1}", (pos[0], pos[1] + 25))

    # Round suit
    if round_suit:
        draw_text(f"Suit: {round_suit.name}", (pos[0], pos[1] + 50))

    # Team 1 score
    draw_text(f"TEAM 1: {sueca_game.team_points[0]}", (pos[0], pos[1] + 75))

    # Team 2 score
    draw_text(f"TEAM 2: {sueca_game.team_points[1]}", (pos[0], pos[1] + 100))

    if error_str:
        draw_text(error_str, (pos[0], pos[1] + 125), (50, 50, 230))

def draw_winner(img, sueca_game : SuecaGame, card_center_labels, pos : Tuple[int, int]):
    contours = [x[3] for x in card_center_labels]
    contours = [c for i, c in enumerate(contours) if i % 2 == sueca_game.winner()]

    text = "TIE"
    COLOR = (0, 0, 255)
    if sueca_game.winner() is not None:
        text = f"TEAM {sueca_game.winner()+1} WINS"
        COLOR = (85, 135, 0)

    cv2.putText(
        img=img, 
        text=text,
        org=pos,
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.8,
        color=COLOR,
        thickness=4,
        lineType=cv2.LINE_AA
    )

    cv2.drawContours(img, contours, -1, (0, 255, 255), 2)
