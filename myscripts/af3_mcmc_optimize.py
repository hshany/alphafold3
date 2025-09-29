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
    --steps 100 --half_life 50 --T_init 0.5 --mutation_rate 1 \
    --fixed_positions 1,5,-1
"""

from __future__ import annotations

import argparse
import os
import sys
import pathlib
from dataclasses import dataclass
import logging
from typing import Dict, Optional, Sequence
import csv


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
    propose_mutation as _propose_mutation,
    seq_idx_to_str as _seq_idx_to_str,
)
from alphafold3.common import folding_input as _folding_input
import dataclasses as _dc
import random as _random


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
        mutation_bias: str = "uniform",  # 'uniform' or 'peptide_plddt'
        random_init_len: Optional[int] = None,
        random_init_seed: Optional[int] = None,
    ) -> None:
        self.json_path = pathlib.Path(json_path)
        self.model_dir = pathlib.Path(model_dir)
        self.peptide_chain_id = peptide_chain_id
        self.receptor_chain_id = receptor_chain_id
        self.num_diffusion_samples = int(num_diffusion_samples)
        self.num_recycles = int(num_recycles)
        self.loss_cfg = loss_cfg or LossConfig()
        self.output_dir = pathlib.Path(output_dir) if output_dir else None
        self.mutation_bias = mutation_bias

        # Stash random init controls
        self._random_init_len = int(random_init_len) if random_init_len else None
        self._random_init_seed = int(random_init_seed) if random_init_seed is not None else None

        # Setup base system
        self.fold_input = load_fold_input(self.json_path)
        # Preserve original peptide sequence (before any randomization)
        try:
            self.original_peptide_seq = self._get_chain_sequence(self.fold_input, self.peptide_chain_id)
        except Exception:
            self.original_peptide_seq = None

        # Apply random peptide initialization before any setup if requested
        rand_seq: Optional[str] = None
        if self._random_init_len is not None and self._random_init_len > 0:
            self.fold_input, rand_seq = self._fold_input_with_random_peptide(
                self.fold_input, self.peptide_chain_id, self._random_init_len, self._random_init_seed
            )

        self.runner = OptimizedPeptideRunner(
            self.model_dir,
            num_diffusion_samples=self.num_diffusion_samples,
            num_recycles=self.num_recycles,
        )
        self.runner.setup_base_system(self.fold_input, self.peptide_chain_id)

        # Cache for evaluated sequences: seq -> {loss, metrics, inference_results}
        self._cache: Dict[str, Dict] = {}

        # Determine initial sequence from fold_input or the random replacement
        self.initial_seq = rand_seq or self._get_chain_sequence(self.fold_input, self.peptide_chain_id)

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

        # Optional per-position mutation bias from peptide residue pLDDT
        pos_conf: Optional[Sequence[float]] = None
        if self.mutation_bias == 'peptide_plddt':
            try:
                # Use best-ranked sample for per-residue stats
                best = max(inf_results, key=lambda r: float(r.metadata.get('ranking_score', float('nan'))))
                atom_chain_ids = best.predicted_structure.chain_id
                atom_b = best.predicted_structure.atom_b_factor  # pLDDT per atom, 0..100
                atom_res_ids = best.predicted_structure.res_id
                import numpy as _np
                # Collect per-residue b-factors for the peptide chain
                mask = _np.asarray(atom_chain_ids) == self.peptide_chain_id
                pep_res_ids = _np.asarray(atom_res_ids)[mask]
                pep_b = _np.asarray(atom_b)[mask]
                # Order by residue id and average per residue
                if pep_res_ids.size > 0:
                    order = _np.argsort(pep_res_ids)
                    res_ids_sorted = pep_res_ids[order]
                    b_sorted = pep_b[order]
                    uniq_ids, starts = _np.unique(res_ids_sorted, return_index=True)
                    # Compute mean per residue id
                    means = []
                    for i, start in enumerate(starts):
                        end = starts[i + 1] if i + 1 < len(starts) else len(b_sorted)
                        means.append(float(_np.nanmean(b_sorted[start:end])))
                    # Map to peptide length by order
                    pep_len = len(peptide_sequence)
                    # If mismatch, pad/truncate
                    if len(means) < pep_len:
                        means = means + [float('nan')] * (pep_len - len(means))
                    elif len(means) > pep_len:
                        means = means[:pep_len]
                    # Convert to [0,1]
                    plddt01 = [min(max(m / 100.0, 0.0), 1.0) if m == m else float('nan') for m in means]
                    # The MCMC skeleton uses weights ~ (1 - pos_confidence).
                    # For probability proportional to (1 - pLDDT), set pos_confidence = pLDDT.
                    pos_conf = plddt01
            except Exception:
                pos_conf = None

        out = {
            'loss': float(loss),
            'metrics': metrics,
            'pos_confidence': pos_conf,  # used if mutation_bias == 'peptide_plddt'
            'inference_results': inf_results,
            'seed': run_out.get('seed'),
        }
        self._cache[peptide_sequence] = out
        return out

    @staticmethod
    def _make_query_only_a3m(seq: str) -> str:
        return f">query\n{seq}\n"

    @staticmethod
    def _fold_input_with_random_peptide(
        base_input: _folding_input.Input,
        peptide_chain_id: str,
        length: int,
        seed: Optional[int] = None,
    ) -> tuple[_folding_input.Input, str]:
        aa = "ACDEFGHIKLMNPQRSTVWY"
        rng = _random.Random(seed)
        rand_seq = "".join(rng.choice(aa) for _ in range(int(length)))

        new_chains = []
        for ch in base_input.chains:
            if isinstance(ch, _folding_input.ProteinChain) and ch.id == peptide_chain_id:
                new_chains.append(
                    _folding_input.ProteinChain(
                        id=ch.id,
                        sequence=rand_seq,
                        ptms=ch.ptms,
                        paired_msa=AF3MCMCEvaluator._make_query_only_a3m(rand_seq),
                        unpaired_msa=AF3MCMCEvaluator._make_query_only_a3m(rand_seq),
                        templates=ch.templates,
                    )
                )
            else:
                new_chains.append(ch)
        return _dc.replace(base_input, chains=tuple(new_chains)), rand_seq


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
    p.add_argument('--random_init_len', type=int, default=None,
                   help='If set, initialize with a random peptide of this length and replace the peptide chain sequence + single-sequence MSA before setup.')
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--half_life', type=float, default=50.0)
    p.add_argument('--T_init', type=float, default=0.5)
    p.add_argument('--mutation_rate', type=int, default=1)
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--fixed_positions', type=str, default=None,
                   help='Comma-separated list of positions to freeze (no mutation). Use 1-based indices from N-terminus (e.g., 1,5,10) or negative indices from C-terminus (e.g., -1,-2 for last and second-last). Fixed positions keep their identity from the fold-input peptide chain; with --random_init_len they are reset to the fold-input residues before starting MCMC.')

    # Loss configuration
    p.add_argument('--loss_mode', type=str, choices=['neg_ranking', 'composite'], default='neg_ranking')
    p.add_argument('--w_rank', type=float, default=1.0)
    p.add_argument('--w_plddt', type=float, default=0.0)
    p.add_argument('--w_iptm', type=float, default=0.0)
    p.add_argument('--w_pae', type=float, default=0.0)
    # Mutation biasing
    p.add_argument('--mutation_bias', type=str, choices=['uniform', 'peptide_plddt'], default='uniform',
                   help='If peptide_plddt, bias mutation probability proportional to (1 - per-residue peptide pLDDT)')

    # Resume option
    p.add_argument('--resume', action='store_true',
                   help='Resume from existing CSVs in output_dir. Uses the last accepted sequence from mcmc_trace.csv as the starting point, continues step/eval numbering, and appends to existing CSVs.')

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
        mutation_bias=args.mutation_bias,
        random_init_len=args.random_init_len,
        random_init_seed=args.seed,
    )

    if args.random_init_len is not None and args.random_init_len > 0:
        if args.init_seq:
            log.warning("random_init_len provided; overriding --init_seq with random initialization.")
        init_seq = evaluator.initial_seq
        log.info("Random initialization enabled: length=%d seed=%s", args.random_init_len, str(args.seed))
    else:
        init_seq = args.init_seq or evaluator.initial_seq

    # Prepare CSV paths before potential resume logic
    summary_csv = args.output_dir / "mcmc_designs.csv"
    trace_csv = args.output_dir / "mcmc_trace.csv"

    # Resume support: determine init_seq, iteration offset, and step offset from existing CSVs
    step_offset = 0
    eval_offset = 0
    prev_best_seq = None
    if args.resume:
        # Compute existing eval count from designs CSV
        if summary_csv.exists():
            try:
                with open(summary_csv, 'r', newline='') as f:
                    r = csv.reader(f)
                    row_count = sum(1 for _ in r)
                eval_offset = max(row_count - 1, 0)  # minus header
            except Exception as e:
                log.warning("Failed to read existing designs CSV for resume: %s", e)
        # Determine step offset and restart seq from trace CSV if available
        if trace_csv.exists():
            try:
                with open(trace_csv, 'r', newline='') as f:
                    r = csv.reader(f)
                    header = next(r, None)
                    rows = list(r)
                if rows:
                    last = rows[-1]
                    try:
                        step_offset = int(last[0]) + 1
                    except Exception:
                        step_offset = 0
                    # columns: ... current_seq_after (idx 11), best_seq (idx 12)
                    last_current_after = last[11]
                    prev_best_seq = last[12]
                    # Choose last accepted current sequence if available; else fall back to last row's current_seq_after
                    chosen = None
                    for r in reversed(rows):
                        try:
                            if int(r[1]) == 1:
                                chosen = r
                                break
                        except Exception:
                            continue
                    init_seq = (chosen[11] if chosen is not None else last_current_after) or init_seq
            except Exception as e:
                log.warning("Failed to read trace CSV for resume: %s", e)
        else:
            # No trace: use designs CSV for starting sequence selection
            if summary_csv.exists():
                try:
                    with open(summary_csv, 'r', newline='') as f:
                        dr = csv.DictReader(f)
                        rows = list(dr)
                    if rows:
                        # Without trace we cannot know acceptance; best approximation is last sequence row
                        init_seq = rows[-1].get('sequence') or init_seq
                except Exception as e:
                    log.warning("Failed to read designs CSV for resume fallback: %s", e)
            else:
                log.warning("--resume set but no CSVs found in %s; starting fresh.", args.output_dir)
    # Resolve fixed positions after determining initial sequence
    fixed_positions_0b = None
    if args.fixed_positions:
        def _parse_fixed_positions(spec: str, L: int) -> list[int]:
            out: set[int] = set()
            for tok in spec.replace(' ', '').split(','):
                if not tok:
                    continue
                try:
                    n = int(tok)
                except ValueError:
                    log.warning("Ignoring non-integer fixed position token: %r", tok)
                    continue
                if n == 0:
                    log.warning("Ignoring zero index in --fixed_positions (use 1-based or negative).")
                    continue
                if n > 0:
                    idx = n - 1
                else:
                    idx = L + n  # e.g., -1 -> L-1
                if 0 <= idx < L:
                    out.add(idx)
                else:
                    log.warning("Ignoring out-of-range fixed position %d for length %d", n, L)
            return sorted(out)

        fixed_positions_0b = _parse_fixed_positions(args.fixed_positions, len(init_seq))
        log.info("Fixed positions (0-based): %s", ",".join(str(i) for i in fixed_positions_0b) if fixed_positions_0b else "<none>")

        # If using random init, reset fixed sites to fold-input identities
        if (args.random_init_len is not None and args.random_init_len > 0) and fixed_positions_0b:
            ref_seq = getattr(evaluator, 'original_peptide_seq', None)
            if ref_seq is None:
                log.warning("Original peptide sequence unavailable; cannot reset fixed positions to fold-input identities.")
            else:
                L = len(init_seq)
                if len(ref_seq) != L:
                    log.warning(
                        "Fold-input peptide length (%d) != random init length (%d); applying fixed identities only where in range.",
                        len(ref_seq), L
                    )
                init_list = list(init_seq)
                for i in fixed_positions_0b:
                    if 0 <= i < min(len(ref_seq), L):
                        init_list[i] = ref_seq[i]
                init_seq = "".join(init_list)
                log.info("Applied fold-input identities at fixed positions to initial sequence.")
    log.info(
        "MCMC setup: steps=%d, half_life=%.3g, T_init=%.3g, mutation_rate=%d, seed=%s, init_len=%d",
        args.steps, args.half_life, args.T_init, args.mutation_rate, str(args.seed), len(init_seq)
    )
    log.info("Loss mode: %s (w_rank=%.3g, w_plddt=%.3g, w_iptm=%.3g, w_pae=%.3g)",
             loss_cfg.mode, loss_cfg.w_rank, loss_cfg.w_plddt, loss_cfg.w_iptm, loss_cfg.w_pae)
    log.info("Mutation bias: %s", args.mutation_bias)

    # Adapter for MCMC skeleton
    # Prepare CSV summary file that will be updated on-the-fly
    if not summary_csv.exists():
        with open(summary_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'eval_idx', 'sequence', 'seed',
                'ranking_score', 'iptm_pep_vs_rec', 'plddt_pep_mean', 'pae_pep_mean',
                'loss', 'variant_dir'
            ])

    # Set up per-step trace logging without modifying the per-sequence CSV
    if not trace_csv.exists():
        with open(trace_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'step', 'accepted', 'T', 'current_loss', 'proposal_loss', 'delta',
                'current_loss_after', 'best_loss', 'best_step',
                'current_seq_before', 'proposal_seq', 'current_seq_after', 'best_seq'
            ])

    eval_counter = {'n': int(eval_offset)}

    def _write_outputs_for_sequence(seq: str, cache_entry: Dict) -> None:
        """Write AF3 outputs for this evaluated sequence and append CSV row.

        Uses the best-ranked sample from `inference_results` for metrics.
        """
        eval_counter['n'] += 1
        idx = eval_counter['n']

        # Create a deterministic variant directory per evaluation
        variant_dir = args.output_dir / f"iter_{idx:04d}"
        job_name = f"iter_{idx:04d}"

        # Write detailed outputs (model files, confidences, ranking CSV)
        try:
            write_variant_outputs(
                inference_results=cache_entry['inference_results'],
                variant_dir=variant_dir,
                job_name=job_name,
                seed=int(cache_entry.get('seed', 0) or 0),
                write_individual_samples=False,
                tgz_confidences=True,
            )
        except Exception as e:
            log.warning(f"Failed to write outputs for eval {idx}: {e}")

        # Metrics and CSV row
        metrics = cache_entry.get('metrics', {})
        row = [
            idx,
            seq,
            int(cache_entry.get('seed', 0) or 0),
            metrics.get('ranking_score'),
            metrics.get('iptm'),
            metrics.get('plddt'),
            metrics.get('pae'),
            float(cache_entry.get('loss', float('nan'))),
            variant_dir.as_posix(),
        ]
        try:
            with open(summary_csv, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)
        except Exception as e:
            log.warning(f"Failed to append CSV for eval {idx}: {e}")

    def evaluate_fn(seq: str) -> Dict[str, float]:
        out = evaluator.evaluate(seq)
        # Write outputs and append to CSV the first time we see this sequence
        # Avoid duplicating work for cached sequences
        cached = evaluator._cache.get(seq, {})
        if cached is out and 'written' not in cached:
            _write_outputs_for_sequence(seq, out)
            cached['written'] = True
        # Return only the required keys to MCMC; keep metrics cached internally
        ret = {'loss': float(out['loss'])}
        if args.mutation_bias == 'peptide_plddt' and out.get('pos_confidence') is not None:
            ret['pos_confidence'] = out['pos_confidence']
        # pos_confidence optional; omit for now to use uniform proposal weights
        return ret

    cfg = MCMCConfig(
        steps=args.steps,
        half_life=args.half_life,
        T_init=args.T_init,
        mutation_rate=args.mutation_rate,
        seed=args.seed,
    )

    # Shared state for tracing sequences across propose/evaluate/progress
    trace_state = {
        'last_current_before': None,
        'last_proposal': None,
        'best_seq': prev_best_seq,
        'init_seq': init_seq,
    }

    def propose_fn_with_trace(seq_idx, mutation_rate, pos_weights, rng):
        # capture current sequence before proposing
        trace_state['last_current_before'] = _seq_idx_to_str(seq_idx)
        mut_idx = _propose_mutation(seq_idx, mutation_rate=mutation_rate, pos_weights=pos_weights, rng=rng)
        trace_state['last_proposal'] = _seq_idx_to_str(mut_idx)
        return mut_idx

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

        # Derive sequences for trace row
        step = int(info.get('step', -1))
        accepted = bool(info.get('accepted'))
        current_before = trace_state['last_current_before'] if step > 0 else trace_state['init_seq']
        proposal_seq = trace_state['last_proposal'] if step > 0 else trace_state['init_seq']
        if accepted:
            current_after = proposal_seq
        else:
            current_after = current_before

        # Track best sequence string by comparing best_step
        if trace_state['best_seq'] is None:
            trace_state['best_seq'] = current_after
        else:
            # If this step established a new best, best_step should equal step
            if int(info.get('best_step', -1)) == step and accepted:
                trace_state['best_seq'] = proposal_seq
            # else unchanged

        # Append to trace CSV
        try:
            with open(trace_csv, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    step,
                    int(accepted),
                    float(info.get('T', float('nan'))),
                    float(info.get('current_loss', float('nan'))),
                    float(info.get('proposal_loss', float('nan'))),
                    float(info.get('delta', float('nan'))),
                    float(info.get('current_loss_after', float('nan'))),
                    float(info.get('best_loss', float('nan'))),
                    int(info.get('best_step', -1)),
                    current_before,
                    proposal_seq,
                    current_after,
                    trace_state['best_seq'],
                ])
        except Exception as e:
            log.warning("Failed to append MCMC trace row: %s", e)

    # Log every 1 step by default to interleave with AF3 logs
    result = run_mcmc_design(
        initial_seq=init_seq,
        evaluate_fn=evaluate_fn,
        config=cfg,
        rng=None,
        propose_fn=propose_fn_with_trace,
        progress_fn=progress_fn,
        log_every=1,
        fixed_positions=fixed_positions_0b,
        step_offset=int(step_offset),
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
        write_individual_samples=False,
        tgz_confidences=True,
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
