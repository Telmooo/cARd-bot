
from typing import List, Optional, Tuple, Union
import numpy as np
import cv2

def draw_grid(images: Union[np.ndarray, List[np.ndarray]], resize: Optional[Tuple[int, int]] = None):
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