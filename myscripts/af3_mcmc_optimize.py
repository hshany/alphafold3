#!/usr/bin/env python3
"""
AF3-backed MCMC sequence optimization
-------------------------------------

This script integrates the AFdesign-style MCMC skeleton with the AlphaFold3
runner implemented in myscripts/peptide_variant_screen.py. It treats AF3 as
the evaluator that returns a scalar loss (configurable) for a given peptide
sequence and optionally per-position confidence to bias mutations.

Current defaults:
- Loss = -ranking_score (maximize AF3 ranking score)
- Mutation bias: uniform over peptide positions (pos_confidence=None)

Example:
  python myscripts/af3_mcmc_optimize.py \
    --json_path path/to/complex.json \
    --model_dir path/to/model \
    --peptide_chain_id C \
    --receptor_chain_id A \
    --steps 100 --half_life 50 --T_init 0.5 --mutation_rate 1
"""

from __future__ import annotations

import argparse
import os
import sys
import pathlib
from dataclasses import dataclass
import logging
from typing import Dict, Optional, Sequence


# Allow importing sibling scripts under myscripts/
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from peptide_variant_screen import (
    OptimizedPeptideRunner,
    load_fold_input,
    write_variant_outputs,
    _compute_best_sample_metrics,
)
from afdesign_mcmc_skeleton import (
    run_mcmc_design,
    MCMCConfig,
)


@dataclass
class LossConfig:
    mode: str = "neg_ranking"  # one of: neg_ranking, composite
    w_rank: float = 1.0         # used in composite: loss -= w_rank * ranking
    w_plddt: float = 0.0        # loss -= w_plddt * (plddt/100)
    w_iptm: float = 0.0         # loss -= w_iptm * iptm
    w_pae: float = 0.0          # loss += w_pae * pae (lower better)


class AF3MCMCEvaluator:
    """Wraps OptimizedPeptideRunner to provide an MCMC evaluator.

    Caches evaluations by sequence to avoid duplicate AF3 runs.
    """

    def __init__(
        self,
        *,
        json_path: pathlib.Path,
        model_dir: pathlib.Path,
        peptide_chain_id: str,
        receptor_chain_id: str,
        num_diffusion_samples: int = 1,
        num_recycles: int = 3,
        loss_cfg: Optional[LossConfig] = None,
        output_dir: Optional[pathlib.Path] = None,
    ) -> None:
        self.json_path = pathlib.Path(json_path)
        self.model_dir = pathlib.Path(model_dir)
        self.peptide_chain_id = peptide_chain_id
        self.receptor_chain_id = receptor_chain_id
        self.num_diffusion_samples = int(num_diffusion_samples)
        self.num_recycles = int(num_recycles)
        self.loss_cfg = loss_cfg or LossConfig()
        self.output_dir = pathlib.Path(output_dir) if output_dir else None

        # Setup base system
        self.fold_input = load_fold_input(self.json_path)
        self.runner = OptimizedPeptideRunner(
            self.model_dir,
            num_diffusion_samples=self.num_diffusion_samples,
            num_recycles=self.num_recycles,
        )
        self.runner.setup_base_system(self.fold_input, self.peptide_chain_id)

        # Cache for evaluated sequences: seq -> {loss, metrics, inference_results}
        self._cache: Dict[str, Dict] = {}

        # Determine initial sequence from fold_input if not provided later
        self.initial_seq = self._get_chain_sequence(self.fold_input, self.peptide_chain_id)

    @staticmethod
    def _get_chain_sequence(fold_input, chain_id: str) -> str:
        for ch in fold_input.chains:
            if getattr(ch, 'id', None) == chain_id and hasattr(ch, 'sequence'):
                return ch.sequence
        raise ValueError(f"Cannot find sequence for chain {chain_id} in fold_input")

    def _compute_loss(self, metrics: Dict[str, float]) -> float:
        mode = self.loss_cfg.mode
        if mode == "neg_ranking":
            ranking = metrics.get('ranking_score', float('nan'))
            if ranking != ranking:  # NaN
                return float('inf')
            return -float(ranking)
        elif mode == "composite":
            # loss = -w_rank*ranking - w_plddt*(plddt/100) - w_iptm*iptm + w_pae*pae
            r = metrics.get('ranking_score', 0.0)
            p = metrics.get('plddt', 0.0)
            i = metrics.get('iptm', 0.0)
            a = metrics.get('pae', 0.0)
            return (
                -self.loss_cfg.w_rank * float(r)
                -self.loss_cfg.w_plddt * (float(p) / 100.0)
                -self.loss_cfg.w_iptm * float(i)
                + self.loss_cfg.w_pae * float(a)
            )
        else:
            raise ValueError(f"Unknown loss mode: {mode}")

    def evaluate(self, peptide_sequence: str) -> Dict[str, float]:
        if peptide_sequence in self._cache:
            return self._cache[peptide_sequence]

        # Run AF3 inference via optimized runner
        run_out = self.runner.run_peptide_variant(peptide_sequence, self.peptide_chain_id)
        inf_results = run_out['inference_results']

        # Compute metrics on best-ranked sample
        metrics = _compute_best_sample_metrics(
            inf_results,
            peptide_chain_id=self.peptide_chain_id,
            receptor_chain_id=self.receptor_chain_id,
        )
        loss = self._compute_loss(metrics)

        # For now, omit per-position confidence (uniform mutation bias)
        out = {
            'loss': float(loss),
            'metrics': metrics,
            # 'pos_confidence': [...],  # optional future enhancement
            'inference_results': inf_results,
            'seed': run_out.get('seed'),
        }
        self._cache[peptide_sequence] = out
        return out


def main() -> None:
    p = argparse.ArgumentParser(description="AF3-backed MCMC sequence optimization")
    p.add_argument('--json_path', type=pathlib.Path, required=True)
    p.add_argument('--model_dir', type=pathlib.Path, required=True)
    p.add_argument('--output_dir', type=pathlib.Path, required=True)
    p.add_argument('--peptide_chain_id', type=str, required=True)
    p.add_argument('--receptor_chain_id', type=str, default='A')

    # AF3 knobs
    p.add_argument('--num_diffusion_samples', type=int, default=1)
    p.add_argument('--num_recycles', type=int, default=3)

    # MCMC knobs
    p.add_argument('--init_seq', type=str, default=None)
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--half_life', type=float, default=50.0)
    p.add_argument('--T_init', type=float, default=0.5)
    p.add_argument('--mutation_rate', type=int, default=1)
    p.add_argument('--seed', type=int, default=None)

    # Loss configuration
    p.add_argument('--loss_mode', type=str, choices=['neg_ranking', 'composite'], default='neg_ranking')
    p.add_argument('--w_rank', type=float, default=1.0)
    p.add_argument('--w_plddt', type=float, default=0.0)
    p.add_argument('--w_iptm', type=float, default=0.0)
    p.add_argument('--w_pae', type=float, default=0.0)

    args = p.parse_args()

    # Configure logging (align with existing logs)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )
    log = logging.getLogger("mcmc")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    loss_cfg = LossConfig(
        mode=args.loss_mode,
        w_rank=args.w_rank,
        w_plddt=args.w_plddt,
        w_iptm=args.w_iptm,
        w_pae=args.w_pae,
    )
    log.info("Initializing AF3 evaluator and compiling model (first run is slower)...")
    evaluator = AF3MCMCEvaluator(
        json_path=args.json_path,
        model_dir=args.model_dir,
        peptide_chain_id=args.peptide_chain_id,
        receptor_chain_id=args.receptor_chain_id,
        num_diffusion_samples=args.num_diffusion_samples,
        num_recycles=args.num_recycles,
        loss_cfg=loss_cfg,
        output_dir=args.output_dir,
    )

    init_seq = args.init_seq or evaluator.initial_seq
    log.info(
        "MCMC setup: steps=%d, half_life=%.3g, T_init=%.3g, mutation_rate=%d, seed=%s, init_len=%d",
        args.steps, args.half_life, args.T_init, args.mutation_rate, str(args.seed), len(init_seq)
    )
    log.info("Loss mode: %s (w_rank=%.3g, w_plddt=%.3g, w_iptm=%.3g, w_pae=%.3g)",
             loss_cfg.mode, loss_cfg.w_rank, loss_cfg.w_plddt, loss_cfg.w_iptm, loss_cfg.w_pae)

    # Adapter for MCMC skeleton
    def evaluate_fn(seq: str) -> Dict[str, float]:
        out = evaluator.evaluate(seq)
        # Return only the required keys to MCMC; keep metrics cached internally
        ret = {'loss': float(out['loss'])}
        # pos_confidence optional; omit for now to use uniform proposal weights
        return ret

    cfg = MCMCConfig(
        steps=args.steps,
        half_life=args.half_life,
        T_init=args.T_init,
        mutation_rate=args.mutation_rate,
        seed=args.seed,
    )

    def progress_fn(info: Dict[str, float]):
        # info keys: step, T, current_loss, proposal_loss, delta, accepted,
        # accepted_total, accept_rate, best_loss, best_step, current_loss_after
        log.info(
            "MCMC step %d | T=%.3g | curr=%.4f prop=%.4f Δ=%.4f %s | best=%.4f@%d | acc_rate=%.2f",
            int(info.get('step', -1)),
            float(info.get('T', 0.0)),
            float(info.get('current_loss', float('nan'))),
            float(info.get('proposal_loss', float('nan'))),
            float(info.get('delta', float('nan'))),
            "ACCEPT" if info.get('accepted') else "reject",
            float(info.get('best_loss', float('nan'))),
            int(info.get('best_step', -1)),
            float(info.get('accept_rate', 0.0)),
        )

    # Log every 1 step by default to interleave with AF3 logs
    result = run_mcmc_design(
        initial_seq=init_seq,
        evaluate_fn=evaluate_fn,
        config=cfg,
        progress_fn=progress_fn,
        log_every=1,
    )

    # Ensure best sequence is evaluated and write AF3 outputs
    best_seq = result.best_seq
    best_eval = evaluator.evaluate(best_seq)
    best_inf = best_eval['inference_results']

    # Write outputs for best design
    job_name = f"mcmc_best"
    variant_dir = args.output_dir / "mcmc_best"
    summary = write_variant_outputs(
        inference_results=best_inf,
        variant_dir=variant_dir,
        job_name=job_name,
        seed=int(best_eval.get('seed', 0) or 0),
    )

    # Save a small text summary
    summary_path = args.output_dir / "mcmc_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(f"Best step: {result.best_step}\n")
        f.write(f"Best loss: {result.best_loss}\n")
        f.write(f"Best seq : {best_seq}\n")
        f.write(f"Ranking score (top): {best_eval['metrics'].get('ranking_score')}\n")
        f.write(f"pLDDT peptide mean (top): {best_eval['metrics'].get('plddt')}\n")
        f.write(f"ipTM peptide vs receptor (top): {best_eval['metrics'].get('iptm')}\n")
        f.write(f"PAE peptide mean (top): {best_eval['metrics'].get('pae')}\n")

    log.info("MCMC complete.")
    log.info("Best step: %d", result.best_step)
    log.info("Best loss: %.4f", result.best_loss)
    log.info("Best seq : %s", best_seq)
    log.info("Outputs written to: %s", variant_dir)
    log.info("Summary: %s", summary_path)


if __name__ == '__main__':
    main()
