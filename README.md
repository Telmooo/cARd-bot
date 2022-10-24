# cARd-bot
"Games are not boring. Games purify our souls and leave room for new development that challenges the mind! They are the products of human wisdom" - Seto Kaiba

**cARd-bot** is an augmented reality application that assists playing card games, namely *Sueca*. Application detects cards played on the table, counts the points of each round and keeps track of team points and game state.

## Setup

**Windows:**
```
cd src/opengl/wheels/python{python version}
pip install PyOpenGL-3.1.6-cp310-cp310-win_amd64.whl
pip install PyOpenGL_accelerate-3.1.6-cp310-cp310-win_amd64.whl

cd ../../../../
pip install -r requirements.txt
```

**Linux (Debian / Ubuntu):**
```
sudo apt-get install freeglut3-dev

pip install -r requirements.txt
pip install pyopengl
pip install PyOpenGL-accelerate (optional)
```

## Usage
Application has three main programs:
- `setup_dataset` - Process a dataset to crop the corner, rank and suit of the cards.
- `calibrate_camera` - Camera calibration via chessboard.
- `card_bot` - Main application, runs the detection and game logic.

## `setup_dataset` specification
```sh
setup_dataset.py [-h] --data DATA --outdir OUTDIR [--split-rank-suit] --regions REGIONS [REGIONS ...]
```

Detailed argument description:
- `-h, --help` - shows help message and usage
- `--data DATA` - path to raw dataset directory to be processed
- `--outdir OUTDIR` - path to output directory where processed dataset is stored
- `--split-rank-suit` - flag to enable splitting of rank and suit into different images. `False` by default
- `--regions REGIONS [REGIONS ...]` - specifies regions of card to crop. May be one of the following combinations:
  - With `split-rank-suit` disabled, specifies the region to crop:
    - `--regions XLIMIT YLIMIT` (equivalent to `--regions 0 0 XLIMIT YLIMIT`)
    - `--regions XBASE YBASE XLIMIT YLIMIT`
  - With `split-rank-suit` enabled, specifies the region of rank to crop and the region of the suit to crop:
    - `--regions XLIMIT YLIMIT YRANKLIMIT` (equivalent to `--regions 0 0 XLIMIT YLIMIT YRANKLIMIT`)
    - `--regions XBASE YBASE XLIMIT YLIMIT YRANKLIMIT` (equivalent to `--regions XBASE YBASE XLIMIT YRANKLIMIT XBASE YRANKLIMIT XLIMIT YLIMIT`)
    - `--regions XRANKBASE YRANKBASE XRANKLIMIT YRANKLIMIT XSUITBASE YSUITBASE XSUITLIMIT YSUITLIMIT`

### Pre-configurated datasets

#### Dataset `cards_normal`
- Cards dimension: **500x726 px**
- Corner (rank + suit): **100x190 px** (no offset)
- Rank: **100x84 px** (no offset)
- Suit: **100x106 px** (offset `Y=84`)

Commands for `setup_dataset`:

**Corner (rank + suit, no split)**
```sh
python src/setup_dataset.py --data ./data/cards_normal --outdir ./data/cards_normal/rank_suit --regions 100 190
```

**Rank + Suit (split)**
```sh
python src/setup_dataset.py --data ./data/cards_normal --outdir ./data/cards_normal/rank_suit --regions 100 190 84 --split-rank-suit
```

#### Dataset `cards_physical`
- Cards dimension: **225x345 px**
- Corner (rank + suit): **45x100 px** (no offset)
- Rank: **45x60 px** (no offset)
- Suit: **45x40 px** (offset `Y=60`)

Commands for `setup_dataset`:

**Corner (rank + suit, no split)**
```sh
python src/setup_dataset.py --data ./data/cards_physical --outdir ./data/cards_physical/rank_suit --regions 45 100
```

**Rank + Suit (split)**
```sh
python src/setup_dataset.py --data ./data/cards_physical --outdir ./data/cards_physical/rank_suit --regions 45 100 60 --split-rank-suit
```

## `calibrate_camera` specification
```sh
calibrate_camera.py [-h] --mode MODE CPOINT [--chess-shape ROWS COLS MM] [--save-frames] [--save-dir SAVE_DIR]
                           [--debug]
```

Detailed argument description:
- `-h, --help` - shows help message and usage
- `--mode MODE CPOINT` - specifies type of acquisition of camera feed and connection point:
  - `--mode usb DEVICENO` - camera feed is acquired directly as a device, and `DEVICENO` specifies the device number in the system
  - `--mode wifi URL` - camera feed is acquired by requesting the URL specified
- `--chess-shape ROWS COLS MM` - specifies the size, in rows and columns, of the chessboard, and the size, in milimeters, of each board square
- `--save-frames` - save captured frames for calibration
- `--save-dir SAVE_DIR` - path to directory to save calibration matrix and saved frames
- `--debug` - enable debug mode

## `card_bot` specification
```sh
card_bot.py [-h] --mode MODE CPOINT --config CONFIG [--debug]
```

Detailed argument description:
- `-h, --help` - shows help message and usage
- `-s S, --trump-suit S` - trump suit for the Sueca game (c - Clubs, d - Diamonds, h - Hearts, s - Spades)
- `--mode MODE CPOINT` - specifies type of acquisition of camera feed and connection point:
  - `--mode usb DEVICENO` - camera feed is acquired directly as a device, and `DEVICENO` specifies the device number in the system
  - `--mode wifi URL` - camera feed is acquired by requesting the URL specified
- `--config CONFIG` - path to `config.yaml` containing program configuration (see example configuration [below](#example-configyaml))
- `--debug` - enable debug mode

### Example `config.yaml`
Example configuration:
```yaml
gaussianBlur:
  kernelSize: 3     # INT. Gaussian blur kernel size. Must be a positive odd number (3, 5, 7, ...)
  sigmaX: 20        # DOUBLE. Gaussian standard deviation in X direction. Must be greater or equal than zero
  sigmaY: 0         # DOUBLE. Gaussian standard deviation in Y direction. Must be greater or equal than zero
clahe:
  clipLimit: 3
  tileGridSize: 2
hysteresis:
  lowerThresh: 150
  highThresh: 200
contours:
  approxPolyDPEpsilon: 0.02
filter:
  slopeThresh: 0.30
card:
  width: 500
  height: 726
  cornerWidth: 100
  cornerHeight: 190
```