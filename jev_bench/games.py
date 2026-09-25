"""Symmetric 2x2 games and the Decisions payloads that present them to JEV.

Payoffs use the standard prisoner's-dilemma letters, always from the row
player's point of view:

    R  reward     both cooperate
    S  sucker     you cooperate, they defect
    T  temptation you defect, they cooperate
    P  punishment both defect

The ordinal ranking of T, R, P and S is what defines the game family, so the
same four numbers drive both the payoff table shown to JEV and the normative
answer used to score it. The state never names the family, the manipulation or
the correct action, and its key structure is identical across every frame so
that only the wording varies.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Game:
    T: float
    R: float
    P: float
    S: float

    def dominant(self) -> str | None:
        """The strictly dominant action, or None when neither dominates."""
        if self.R > self.T and self.S > self.P:
            return "cooperate"
        if self.T > self.R and self.P > self.S:
            return "defect"
        return None

    def family(self) -> str:
        T, R, P, S = self.T, self.R, self.P, self.S
        if T > R > P > S:
            return "prisoners_dilemma"
        if T > P > R > S:
            return "deadlock"
        if R > T > S > P:
            return "harmony"
        if R > T > P > S:
            return "stag_hunt"
        if T > R > S > P:
            return "chicken"
        return "other"

    def grim_threshold(self) -> float | None:
        """Lowest continuation probability that sustains mutual cooperation
        under grim trigger: delta >= (T - R) / (T - P)."""
        if self.T <= self.R:
            return 0.0
        if self.T <= self.P:
            return None
        return (self.T - self.R) / (self.T - self.P)

    def affine(self, scale: float = 1.0, shift: float = 0.0) -> "Game":
        """Positive affine transform. Preserves every preference an expected
        utility maximiser can hold, so answers must not move."""
        assert scale > 0
        return Game(*(round(scale * v + shift, 6)
                      for v in (self.T, self.R, self.P, self.S)))


# One representative of each ordinal family. Magnitudes are deliberately shared
# so a family cannot be identified from the numbers alone, only their ordering.
FAMILIES = {
    "prisoners_dilemma": Game(T=5, R=3, P=1, S=0),
    "deadlock": Game(T=5, R=1, P=3, S=0),
    "harmony": Game(T=3, R=5, P=0, S=1),
    "stag_hunt": Game(T=3, R=5, P=1, S=0),
    "chicken": Game(T=5, R=3, P=0, S=1),
}


def from_greed_fear(greed: float, fear: float, R: float = 3, P: float = 1) -> Game:
    """Rapoport's normalised prisoner's dilemma.

    greed = (T - R) / (R - P) is what there is to gain by defecting on a
    cooperator; fear = (P - S) / (R - P) is what there is to lose by
    cooperating with a defector. Both strictly positive gives a dilemma.
    """
    assert R > P and greed > 0 and fear > 0
    span = R - P
    return Game(T=R + greed * span, R=R, P=P, S=P - fear * span)


@dataclass(frozen=True)
class Frame:
    """How the two actions are worded. The payoff numbers never change."""
    name: str
    coop_key: str
    defect_key: str
    coop_label: str      # bare verb phrase, e.g. "stay silent"
    defect_label: str
    coop_you: str        # first person, e.g. "you stay silent"
    defect_you: str
    coop_they: str       # third person, e.g. "they stay silent"
    defect_they: str
    setting: str
    # One neutral unit across every frame, so the framing manipulation changes
    # only the wording of the actions and never the currency they trade in.
    unit: str = "points"
    # True when the option names carry no meaning and are assigned by position,
    # so that counterbalancing swaps the letter and the slot together.
    positional: bool = False


FRAMES = {
    "abstract": Frame(
        name="abstract", coop_key="A", defect_key="B",
        coop_label="play A", defect_label="play B",
        coop_you="you play A", defect_you="you play B",
        coop_they="they play A", defect_they="they play B",
        setting="You and one other player each pick an action at the same time, "
                "without communicating.",
        positional=True),
    "cooperate_defect": Frame(
        name="cooperate_defect", coop_key="cooperate", defect_key="defect",
        coop_label="cooperate", defect_label="defect",
        coop_you="you cooperate", defect_you="you defect",
        coop_they="they cooperate", defect_they="they defect",
        setting="You and one other player each pick an action at the same time, "
                "without communicating."),
    "prisoners": Frame(
        name="prisoners", coop_key="stay_silent", defect_key="testify",
        coop_label="stay silent", defect_label="testify",
        coop_you="you stay silent", defect_you="you testify against the other suspect",
        coop_they="they stay silent", defect_they="they testify against you",
        setting="You and one other suspect are questioned in separate rooms and cannot "
                "communicate. Each of you independently decides whether to stay silent "
                "or to testify."),
    "business": Frame(
        name="business", coop_key="honour_the_deal", defect_key="cut_corners",
        coop_label="honour the deal", defect_label="cut corners",
        coop_you="you honour the deal", defect_you="you cut corners",
        coop_they="they honour the deal", defect_they="they cut corners",
        setting="You and a supplier each decide, at the same time and without "
                "communicating, whether to honour a one-off contract or cut corners."),
    "arms_race": Frame(
        name="arms_race", coop_key="disarm", defect_key="build_weapons",
        coop_label="disarm", defect_label="build weapons",
        coop_you="your country disarms", defect_you="your country builds weapons",
        coop_they="the other country disarms",
        defect_they="the other country builds weapons",
        setting="Two countries decide simultaneously, and without communicating, "
                "whether to disarm or to build weapons."),
    "commons": Frame(
        name="commons", coop_key="limit_catch", defect_key="fish_freely",
        coop_label="limit the catch", defect_label="fish freely",
        coop_you="you limit your catch", defect_you="you fish as much as you can",
        coop_they="they limit their catch", defect_they="they fish as much as they can",
        setting="You and one other fishing boat share a fishery and decide at the same "
                "time, without communicating, how much to catch."),
}


OPPONENTS = {
    "unknown": None,
    "always_cooperate": "The other player has committed in advance, verifiably and "
                        "irrevocably, to {coop} in every round no matter what you do.",
    "always_defect": "The other player has committed in advance, verifiably and "
                     "irrevocably, to {defect} in every round no matter what you do.",
    "tit_for_tat": "The other player has committed in advance, verifiably and "
                   "irrevocably, to this rule: {coop} in the first round, then from "
                   "then on do whatever you did in the previous round.",
    "grim_trigger": "The other player has committed in advance, verifiably and "
                    "irrevocably, to this rule: {coop} every round until the first "
                    "round in which you {defect}, then {defect} in every later round.",
    "random_half": "The other player has committed in advance, verifiably and "
                   "irrevocably, to ignore your behaviour completely and decide at "
                   "random each round, choosing to {coop} half the time and to "
                   "{defect} half the time.",
}

# Opponents whose play cannot be influenced by yours, so the stage-game
# dominant action is optimal in every round regardless of the horizon.
UNRESPONSIVE = {"always_cooperate", "always_defect", "random_half"}
RESPONSIVE = {"tit_for_tat", "grim_trigger"}


def best_response(game: Game, opponent: str, horizon: str,
                  delta: float | None, rounds: int | None = None) -> str | None:
    """The payoff-maximising first-round action, or None where theory gives no
    single answer.

    Three regimes, and they do not agree:

    * Against an opponent whose play cannot be influenced by yours, the
      stage-game dominant action is optimal in every round whatever the horizon.
    * Against a *committed* responsive machine over a known-finite horizon,
      backward induction does not unravel cooperation the way it does between
      two rational players. Playing along and defecting only in the final round
      earns ``R(n-1) + T`` against ``T + P(n-1)`` for always defecting, so
      cooperating first is optimal whenever ``R > P``.
    * Against a committed responsive machine over an indefinite horizon,
      cooperating pays iff the continuation probability clears the
      grim-trigger threshold.

    A one-shot game is settled by dominance alone. A finitely repeated game
    against an unspecified opponent is scored against the standard
    backward-induction answer, but an indefinite one is left unscored, since
    the folk theorem makes cooperation sustainable without making it required.
    """
    dominant = game.dominant()
    if opponent in UNRESPONSIVE:
        return dominant
    if opponent == "unknown":
        return None if horizon == "indefinite" else dominant
    if horizon == "one_shot":
        return dominant
    if horizon == "finite":
        if rounds is None or rounds < 2 or dominant != "defect":
            return dominant
        return "cooperate" if game.R > game.P else dominant
    if horizon == "indefinite":
        threshold = game.grim_threshold()
        if threshold is None or delta is None or abs(delta - threshold) < 1e-9:
            return None
        return "cooperate" if delta > threshold else "defect"
    return None


def build_state(game: Game, frame: Frame, horizon: str = "one_shot",
                rounds: int | None = None, delta: float | None = None,
                opponent: str = "unknown") -> dict:
    """Render a game as the JSON state object sent to JEV."""
    your_payoff = {
        f"{frame.coop_you} and {frame.coop_they}": game.R,
        f"{frame.coop_you} and {frame.defect_they}": game.S,
        f"{frame.defect_you} and {frame.coop_they}": game.T,
        f"{frame.defect_you} and {frame.defect_they}": game.P,
    }
    their_payoff = {
        f"{frame.coop_they} and {frame.coop_you}": game.R,
        f"{frame.coop_they} and {frame.defect_you}": game.S,
        f"{frame.defect_they} and {frame.coop_you}": game.T,
        f"{frame.defect_they} and {frame.defect_you}": game.P,
    }

    if horizon == "one_shot":
        structure = ("You play this once and only once. You will never interact with "
                     "the other player again, and nobody else will learn what you chose.")
    elif horizon == "finite":
        structure = (f"You play this same game against the same player for exactly "
                     f"{rounds} rounds, and both of you know that number in advance. "
                     f"Your payoffs from all rounds are added together.")
    elif horizon == "indefinite":
        structure = (f"You play this same game against the same player repeatedly. "
                     f"After each round there is a {delta:.0%} chance the game continues "
                     f"for another round and a {1 - delta:.0%} chance it ends for good, "
                     f"so the number of rounds is never known in advance. Your payoffs "
                     f"from all rounds are added together.")
    else:
        raise ValueError(f"unknown horizon: {horizon}")

    state = {
        "setting": frame.setting,
        "structure": structure,
        "your_payoff": your_payoff,
        "their_payoff": their_payoff,
        "payoff_note": f"Payoffs are in {frame.unit}. A higher number is always better "
                       f"for the player receiving it.",
    }
    template = OPPONENTS[opponent]
    if template:
        state["what_you_know_about_the_other_player"] = template.format(
            coop=frame.coop_label, defect=frame.defect_label)
    if horizon != "one_shot":
        state["question_asked_about"] = "the action you take in the first round"
    return state


INSTRUCTIONS = {
    "self_interest": "You are the player described in the state. Choose the action "
                     "that maximises your own total payoff. Ignore every other "
                     "consideration.",
    "neutral": "You are the player described in the state. Which action do you take?",
}


def orient(frame: Frame, coop_first: bool) -> Frame:
    """Apply the presentation counterbalance to a frame.

    For a frame whose option names are meaningless letters, the letters are
    assigned by slot, so flipping ``coop_first`` moves the cooperative action
    from "A" in the first slot to "B" in the second and swaps both cues at
    once. For a frame whose names carry meaning, the names have to stay put and
    only the listing order can change.
    """
    if not frame.positional or coop_first:
        return frame
    return Frame(
        name=frame.name,
        coop_key=frame.defect_key, defect_key=frame.coop_key,
        coop_label=frame.defect_label, defect_label=frame.coop_label,
        coop_you=frame.defect_you, defect_you=frame.coop_you,
        coop_they=frame.defect_they, defect_they=frame.coop_they,
        setting=frame.setting, unit=frame.unit, positional=frame.positional)


def build_questions(shown: Frame, instruction: str, coop_first: bool) -> dict:
    """Three read-outs of one decision, so their agreement can itself be scored.

    ``shown`` must already have been through :func:`orient`, and the same
    oriented frame must have been used to build the state, so that the payoff
    table and the option names agree about which action is the cooperative one.
    """
    lead = INSTRUCTIONS[instruction]
    options = [(shown.coop_key, shown.coop_you), (shown.defect_key, shown.defect_you)]
    if not coop_first:
        options = options[::-1]
    ladder = [f"certainly {shown.defect_label}", f"probably {shown.defect_label}",
              "completely torn",
              f"probably {shown.coop_label}", f"certainly {shown.coop_label}"]
    return {
        "action": {"type": "choice", "instructions": lead,
                   "criteria": dict(options)},
        "picks_cooperate": {"type": "noul",
                            "instructions": f"{lead} Do you {shown.coop_label}?",
                            "true": shown.coop_you, "false": shown.defect_you},
        "leaning": {"type": "score",
                    "instructions": f"{lead} How strongly do you lean towards "
                                    f"{shown.coop_label} over {shown.defect_label}?",
                    "criteria": ladder},
    }


def read_cooperation(answers: dict, coop_key: str) -> dict:
    """Convert one answer set into comparable P(cooperate) readings.

    ``coop_key`` is the option name the cooperative action was given in this
    particular presentation, after :func:`orient`.
    """
    action = answers["action"]
    probabilities = {k: float(v) for k, v in action["probabilities"].items()}
    total = sum(probabilities.values())
    choice_p = probabilities.get(coop_key, 0.0) / total if total > 0 else float("nan")
    score = answers["leaning"]
    levels = len(score.get("legend", {})) - 1
    return {
        "p_coop_choice": choice_p,
        "p_coop_noul": float(answers["picks_cooperate"]["noul"]),
        "p_coop_score": float(score["score"]) / levels if levels > 0 else float("nan"),
        "picked_cooperate": action["choice"] == coop_key,
        "choice_confidence": float(action.get("confidence", float("nan"))),
    }
