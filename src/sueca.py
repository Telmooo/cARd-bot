
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional
import random

class Rank(Enum):
    Ace = 1
    Two = 2
    Three = 3
    Four = 4
    Five = 5
    Six = 6
    Seven = 7
    Eight = 8
    Nine = 9
    Ten = 10
    Queen = 11
    Jack = 12
    King = 13

class Suit(Enum):
    Clubs = "c"
    Diamonds = "d"
    Hearts = "h"
    Spades = "s"

    def is_red(self):
        return self in {Suit.Hearts, Suit.Diamonds}

@dataclass()
class Card:
    rank: Rank
    suit: Suit

@dataclass()
class SuecaRound:
    suit: Suit
    cards: List[Card] # 0, 1, 0, 1

    points_by_rank = {
        Rank.Ace: 11,
        Rank.Seven: 10,
        Rank.King: 4,
        Rank.Jack: 3,
        Rank.Queen: 2,
    }

    def winner(self, trump_suit: Suit) -> int:
        best_card = max(
            self.cards,
            key=lambda c: (0 if c.suit != self.suit and c.suit != trump_suit else
                SuecaRound.points_by_rank.get(c.rank, 0) + 100 * (c.suit == trump_suit))
        )
        return self.cards.index(best_card) % 2

    def points(self) -> int:
        return sum(map(lambda c: SuecaRound.points_by_rank.get(c.rank, 0), self.cards))

class SuecaGame:
    NUM_ROUNDS = 10

    def __init__(self, trump_suit: Suit = None) -> None:
        self.trump_suit = random.choice(list(Suit))
        self.team_points = [0, 0]
        self.rounds_evaluated = 0

    def evaluate_round(self, r: SuecaRound):
        self.team_points[r.winner(self.trump_suit)] += r.points()
        self.rounds_evaluated += 1

    def is_finished(self) -> bool:
        return self.rounds_evaluated == SuecaGame.NUM_ROUNDS

    def is_tied(self) -> bool:
        return self.team_points[0] == self.team_points[1]

    def winner(self) -> Optional[int]:
        if not self.is_finished() or self.is_tied():
            return None
        return self.team_points.index(max(self.team_points))
