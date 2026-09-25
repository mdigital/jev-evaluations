"""The game-theory benchmark: seven blocks, each varying one factor.

Every block shares a common baseline (prisoner's dilemma, abstract wording,
one-shot, unknown opponent, explicit self-interest instruction) and moves one
thing at a time, so an effect can be attributed to the factor that moved.
Every cell is run under both option orderings so position bias cancels in the
aggregate and can still be measured on its own.
"""
from __future__ import annotations

from . import games as G
from .client import DecisionsClient

REPS_MAIN = 8
REPS_SWEEP = 6

# Prisoner's dilemmas whose grim-trigger thresholds are far apart, so the
# folk-theorem prediction moves a long way across the three.
DELTA_SWEEP_GAMES = {
    "low_temptation": G.Game(T=4, R=3, P=1, S=0),    # delta* = 1/3
    "mid_temptation": G.Game(T=5, R=3, P=1, S=0),    # delta* = 1/2
    "high_temptation": G.Game(T=9, R=3, P=1, S=0),   # delta* = 3/4
}
DELTAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]

GREEDS = [0.05, 0.25, 0.5, 1.0, 2.0, 4.0]
FEARS = [0.05, 0.25, 0.5, 1.0, 2.0, 4.0]

# Positive affine transforms. Every one preserves the preferences of an
# expected-utility maximiser, so no answer is allowed to move. "loss_framed"
# is the interesting one: it pushes three of the four payoffs below zero.
AFFINE = {
    "identity": (1.0, 0.0),
    "times_ten": (10.0, 0.0),
    "tenth": (0.1, 0.0),
    "shift_up": (1.0, 100.0),
    "loss_framed": (1.0, -3.0),
    "scale_and_shift": (2.0, 50.0),
}

HORIZONS = [
    ("one_shot", None, None),
    ("finite", 2, None),
    ("finite", 5, None),
    ("finite", 10, None),
    ("finite", 100, None),
    ("indefinite", None, 0.5),
    ("indefinite", None, 0.9),
]


def _trial(block, game, frame, instruction, coop_first, repetition,
           horizon="one_shot", rounds=None, delta=None, opponent="unknown",
           **meta):
    # One oriented frame drives both halves of the request, so the payoff table
    # and the option names can never disagree about which action cooperates.
    shown = G.orient(frame, coop_first)
    state = G.build_state(game, shown, horizon, rounds, delta, opponent)
    questions = G.build_questions(shown, instruction, coop_first)
    payload = {"state": state, "questions": questions}
    trial = {
        "block": block,
        "family": game.family(),
        "T": game.T, "R": game.R, "P": game.P, "S": game.S,
        "frame": frame.name,
        "coop_key": shown.coop_key,
        "instruction": instruction,
        "coop_first": coop_first,
        "horizon": horizon,
        "rounds": rounds,
        "delta": delta,
        "opponent": opponent,
        "repetition": repetition,
        "dominant": game.dominant(),
        "best_response": G.best_response(game, opponent, horizon, delta, rounds),
        "grim_threshold": game.grim_threshold(),
        "payload": payload,
        "record_key": DecisionsClient.record_key(payload, repetition),
    }
    trial.update(meta)
    return trial


def build_trials() -> list[dict]:
    abstract = G.FRAMES["abstract"]
    pd = G.FAMILIES["prisoners_dilemma"]
    harmony = G.FAMILIES["harmony"]
    trials: list[dict] = []

    # 1. Does the answer track the ordinal structure of the game?
    for family, game in G.FAMILIES.items():
        for instruction in ("self_interest", "neutral"):
            for coop_first in (True, False):
                for rep in range(REPS_MAIN):
                    trials.append(_trial("family", game, abstract, instruction,
                                         coop_first, rep))

    # 2. Does it track how much there is to gain and to lose, within one family?
    for greed in GREEDS:
        for fear in FEARS:
            game = G.from_greed_fear(greed, fear)
            for coop_first in (True, False):
                for rep in range(REPS_SWEEP):
                    trials.append(_trial("greed_fear", game, abstract,
                                         "self_interest", coop_first, rep,
                                         greed=greed, fear=fear))

    # 3. Does cooperation switch on near the continuation probability that
    #    theory says makes it pay? Against tit-for-tat the threshold is a sharp
    #    decision boundary; against an unknown opponent only the shape is
    #    predicted, so those cells stay unscored.
    for label, game in DELTA_SWEEP_GAMES.items():
        for delta in DELTAS:
            for opponent in ("tit_for_tat", "unknown"):
                for coop_first in (True, False):
                    for rep in range(REPS_SWEEP):
                        trials.append(_trial("delta_sweep", game, abstract,
                                             "self_interest", coop_first, rep,
                                             horizon="indefinite", delta=delta,
                                             opponent=opponent, payoff_set=label))

    # 4. Does a known finite horizon read differently from an open-ended one?
    for horizon, rounds, delta in HORIZONS:
        for opponent in ("unknown", "tit_for_tat"):
            for coop_first in (True, False):
                for rep in range(REPS_MAIN):
                    trials.append(_trial("horizon", pd, abstract, "self_interest",
                                         coop_first, rep, horizon=horizon,
                                         rounds=rounds, delta=delta,
                                         opponent=opponent))

    # 5. Does it best-respond to an opponent that has already committed?
    for opponent in G.OPPONENTS:
        for horizon, delta in (("one_shot", None), ("indefinite", 0.9)):
            for coop_first in (True, False):
                for rep in range(REPS_MAIN):
                    trials.append(_trial("opponent", pd, abstract, "self_interest",
                                         coop_first, rep, horizon=horizon,
                                         delta=delta, opponent=opponent))

    # 6. Does the wording of the actions move an answer it must not move?
    for frame in G.FRAMES.values():
        for game in (pd, harmony):
            for coop_first in (True, False):
                for rep in range(REPS_MAIN):
                    trials.append(_trial("framing", game, frame, "self_interest",
                                         coop_first, rep))

    # 7. Does rescaling or shifting the payoffs move it? It must not.
    for name, (scale, shift) in AFFINE.items():
        for game in (pd, harmony):
            transformed = game.affine(scale, shift)
            for coop_first in (True, False):
                for rep in range(REPS_MAIN):
                    trials.append(_trial("affine", transformed, abstract,
                                         "self_interest", coop_first, rep,
                                         transform=name, base_family=game.family()))

    for index, trial in enumerate(trials):
        trial["trial_id"] = f"{trial['block']}-{index:06d}"
    return trials
