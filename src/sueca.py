
from dataclasses import dataclass
from enum import Enum, auto
from functools import total_ordering
from typing import List

class Rank(Enum):
    Ace = auto()
    Two = auto()
    Three = auto()
    Four = auto()
    Five = auto()
    Six = auto()
    Seven = auto()
    Jack = auto()
    Queen = auto()
    King = auto()

class Suit(Enum):
    Clubs = auto()
    Diamonds = auto()
    Hearts = auto()
    Spades = auto()

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
            key=lambda c: SuecaRound.points_by_rank[c.rank] + 100 if c.suit == trump_suit else 0
        )
        winner = self.cards.index(best_card)
        return winner % 2

    def points(self) -> int:
        return sum(map(lambda c: SuecaRound.points_by_rank.get(c.rank, 0), self.cards))

class SuecaGame:
    def __init__(self, trump_suit: Suit) -> None:
        self.trump_suit = trump_suit
        self.team_points = [0, 0]

    def evaluate_round(self, r: SuecaRound):
        self.team_points[r.winner(self.trump_suit)] += r.points()