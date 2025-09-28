"""
Minimal AFdesign-style MCMC skeleton
------------------------------------

This script replicates the core AFdesign MCMC workflow found in
colabdesign.af.design._design_mcmc, but abstracts away evaluation so you
can plug in any scoring function (e.g., Alphafold3-based evaluation later).

Key features:
- Simulated annealing temperature schedule (exponential half-life decay)
- Metropolis acceptance criterion on scalar loss
- Simple mutation proposals with optional position weights
- Records best sequence and simple history

Usage:
- Import `run_mcmc_design` and pass an `evaluate_fn` that returns a dict
  with at least a float `loss`. Optionally return `pos_confidence` to bias
  mutation positions (e.g., AF pLDDT -> higher mutation probability where
  confidence is lower).

The implementation intentionally avoids any AF/colabdesign dependencies.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple


# 20 standard amino acids (order can be adjusted as needed)
AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_ALPHABET)}
IDX_TO_AA = {i: aa for i, aa in enumerate(AA_ALPHABET)}


def seq_str_to_idx(seq: str) -> List[int]:
    return [AA_TO_IDX[a] for a in seq]


def seq_idx_to_str(seq_idx: Sequence[int]) -> str:
    return "".join(IDX_TO_AA[i] for i in seq_idx)


def default_position_weights(
    length: int, pos_confidence: Optional[Sequence[float]] = None
) -> List[float]:
    """
    Create mutation position weights.
    - If `pos_confidence` provided (e.g., pLDDT in [0,1]), use (1 - confidence).
    - Else, uniform over positions.
    """
    if pos_confidence is None:
        return [1.0] * length
    w = []
    for c in pos_confidence:
        if c is None or math.isnan(c):
            w.append(0.0)
        else:
            w.append(max(1.0 - float(c), 0.0))
    s = sum(w)
    if s <= 0:
        # fall back to uniform if all zero
        return [1.0] * length
    return [x / s for x in w]


def propose_mutation(
    seq_idx: Sequence[int],
    mutation_rate: int = 1,
    pos_weights: Optional[Sequence[float]] = None,
    rng: random.Random | None = None,
) -> List[int]:
    """
    Propose a mutated sequence by applying `mutation_rate` single-residue changes.
    - Position selection follows `pos_weights` if provided; else uniform.
    - New residue is sampled uniformly from 19 alternatives (not equal to current).
    """
    if rng is None:
        rng = random
    out = list(seq_idx)
    L = len(out)

    # Prepare cumulative weights for efficient sampling
    if pos_weights is None:
        cum = None
    else:
        cum = []
        s = 0.0
        for w in pos_weights:
            s += float(w)
            cum.append(s)
        # Normalize cumulative sum to 1.0 for safety
        if s > 0:
            cum = [x / s for x in cum]
        else:
            cum = None

    def draw_position() -> int:
        if cum is None:
            return rng.randrange(L)
        u = rng.random()
        lo, hi = 0, L - 1
        # binary search over cumulative
        while lo < hi:
            mid = (lo + hi) // 2
            if u <= cum[mid]:
                hi = mid
            else:
                lo = mid + 1
        return lo

    for _ in range(max(1, int(mutation_rate))):
        i = draw_position()
        current = out[i]
        # sample a different amino acid uniformly
        choices = list(range(len(AA_ALPHABET)))
        choices.remove(current)
        out[i] = rng.choice(choices)

    return out


@dataclass
class MCMCConfig:
    steps: int = 1000
    half_life: float = 200.0  # temperature decays by 0.5 every `half_life` steps
    T_init: float = 0.01
    mutation_rate: int = 1
    seed: Optional[int] = None


@dataclass
class MCMCResult:
    best_seq: str
    best_loss: float
    best_step: int
    history: List[Dict]


def default_temperature(step: int, T_init: float, half_life: float) -> float:
    # Matches: T_init * (exp(log(0.5) / half_life) ** step)
    return T_init * (0.5 ** (step / max(1e-8, half_life)))


def run_mcmc_design(
    initial_seq: str,
    evaluate_fn: Callable[[str], Dict[str, float]],
    config: MCMCConfig | None = None,
    rng: random.Random | None = None,
    propose_fn: Callable[[Sequence[int], int, Optional[Sequence[float]], Optional[random.Random]], List[int]] = propose_mutation,
    progress_fn: Optional[Callable[[Dict[str, float]], None]] = None,
    log_every: int = 0,
) -> MCMCResult:
    """
    Run MCMC with simulated annealing on sequences.

    Args:
      initial_seq: Starting amino-acid sequence string (alphabet defined above).
      evaluate_fn: Callable that takes a sequence string and returns a dict with:
        - 'loss' (float): required. Lower is better.
        - 'pos_confidence' (optional Sequence[float]): per-position confidence in [0,1].
      config: MCMCConfig hyperparameters.
      rng: optional random.Random instance for reproducibility beyond config.seed.

    Returns:
      MCMCResult with best sequence, loss, step, and a simple history.
    """
    if config is None:
        config = MCMCConfig()
    if rng is None:
        rng = random.Random(config.seed) if config.seed is not None else random
    elif config.seed is not None:
        rng.seed(config.seed)

    # state
    current_seq_idx = seq_str_to_idx(initial_seq)
    current_seq = initial_seq
    current_eval = evaluate_fn(current_seq)
    current_loss = float(current_eval["loss"])  # may be inf on first iteration in AFdesign
    pos_conf = current_eval.get("pos_confidence")

    best_seq = current_seq
    best_loss = current_loss
    best_step = 0

    history: List[Dict] = []

    accepts = 0
    for step in range(config.steps):
        T = default_temperature(step, config.T_init, config.half_life)

        # Prepare mutation position weights from confidence (optional)
        pos_w = default_position_weights(len(current_seq_idx), pos_conf)

        # Propose mutation(s)
        if step == 0:
            mut_seq_idx = list(current_seq_idx)
        else:
            mut_seq_idx = propose_fn(
                current_seq_idx,
                mutation_rate=config.mutation_rate,
                pos_weights=pos_w,
                rng=rng,
            )

        mut_seq = seq_idx_to_str(mut_seq_idx)
        mut_eval = evaluate_fn(mut_seq)
        mut_loss = float(mut_eval["loss"])

        # Metropolis acceptance
        delta = mut_loss - current_loss
        accept = (step == 0) or (delta < 0.0) or (rng.random() < math.exp(-delta / max(1e-12, T)))

        # Log proposal before potentially updating state
        step_info = {
            "step": step,
            "T": T,
            "current_loss": current_loss,
            "proposal_loss": mut_loss,
            "delta": delta,
            "accepted": bool(accept),
            "best_loss": best_loss,
        }
        history.append(step_info)

        if accept:
            accepts += 1
            current_seq_idx = mut_seq_idx
            current_seq = mut_seq
            current_loss = mut_loss
            pos_conf = mut_eval.get("pos_confidence")

            if current_loss < best_loss:
                best_loss = current_loss
                best_seq = current_seq
                best_step = step

        # Emit progress
        if progress_fn is not None and (log_every and ((step + 1) % log_every == 0) or (step == config.steps - 1)):
            acc_rate = accepts / float(step + 1)
            emit = dict(step_info)
            emit.update({
                "accepted_total": accepts,
                "accept_rate": acc_rate,
                "best_loss": best_loss,
                "best_step": best_step,
                "current_loss_after": current_loss,
            })
            try:
                progress_fn(emit)
            except Exception:
                pass

    return MCMCResult(
        best_seq=best_seq, best_loss=best_loss, best_step=best_step, history=history
    )


# ---------------------------------------------------------------------------
# Example usage with a dummy evaluator
# ---------------------------------------------------------------------------

def _dummy_evaluate_factory(target: str) -> Callable[[str], Dict[str, float]]:
    """
    Create a toy evaluator that scores sequences by Hamming distance to `target`.
    Also returns a fake per-position confidence inversely related to match.
    """
    target_idx = seq_str_to_idx(target)

    def evaluate(seq: str) -> Dict[str, float]:
        s_idx = seq_str_to_idx(seq)
        # Hamming distance as loss
        loss = sum(int(a != b) for a, b in zip(s_idx, target_idx))
        # Fake confidence: 1.0 where matches target, 0.3 otherwise
        pos_conf = [1.0 if a == b else 0.3 for a, b in zip(s_idx, target_idx)]
        return {"loss": float(loss), "pos_confidence": pos_conf}

    return evaluate


if __name__ == "__main__":
    # Demonstration: converge to the target sequence using MCMC
    target = "ACDEFGHIKLMN"  # any sequence from AA_ALPHABET
    init = "NMLKIHGFEDCA"    # start far from target

    cfg = MCMCConfig(steps=500, half_life=200.0, T_init=0.5, mutation_rate=1, seed=0)
    evaluator = _dummy_evaluate_factory(target)

    result = run_mcmc_design(initial_seq=init, evaluate_fn=evaluator, config=cfg)
    print("Best step:", result.best_step)
    print("Best loss:", result.best_loss)
    print("Best seq :", result.best_seq)
