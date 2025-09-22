#!/usr/bin/env python3
"""
Simple featurisation benchmark that leverages library timing.

- Calls alphafold3.data.featurisation.featurise_input directly.
- Per-step timings are emitted by the pipeline when AF3_FEAT_TIMING=1.
- Prints only total wall time per run here.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import time
from typing import Iterable

import numpy as np

import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, (REPO_ROOT / 'src').as_posix())

from alphafold3.common import folding_input
from alphafold3.constants import chemical_components
from alphafold3.data import featurisation
from absl import logging as absl_logging


def _load_fold_input(path: pathlib.Path) -> folding_input.Input:
    with open(path, 'rt') as f:
        return folding_input.Input.from_json(f.read(), path)


def _make_query_only_a3m(seq: str) -> str:
    return f">query\n{seq}\n"


def _override_peptide_chain(
    base: folding_input.Input,
    chain_id: str,
    sequence: str,
) -> folding_input.Input:
    new_chains = []
    for ch in base.chains:
        if ch.id == chain_id and isinstance(ch, folding_input.ProteinChain):
            new_chains.append(
                folding_input.ProteinChain(
                    id=ch.id,
                    sequence=sequence,
                    ptms=ch.ptms,
                    paired_msa=_make_query_only_a3m(sequence),
                    unpaired_msa=_make_query_only_a3m(sequence),
                    templates=ch.templates,
                )
            )
        else:
            new_chains.append(ch)
    return base.__class__(
        name=base.name,
        chains=tuple(new_chains),
        rng_seeds=tuple(base.rng_seeds),
        bonded_atom_pairs=base.bonded_atom_pairs,
        user_ccd=base.user_ccd,
        output_dir=getattr(base, 'output_dir', ''),
    )


def _parse_buckets(s: str | None) -> list[int] | None:
    if not s:
        return None
    return [int(x.strip()) for x in s.split(',') if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', type=pathlib.Path, required=True, help='Path to fold_input.json')
    ap.add_argument('--peptide_chain_id', type=str, default=None)
    ap.add_argument('--peptide', type=str, default=None)
    ap.add_argument('--runs', type=int, default=1)
    ap.add_argument('--warmup', type=int, default=0, help='Warmup iterations')
    ap.add_argument('--buckets', type=str, default=None, help='Comma-separated bucket sizes')
    ap.add_argument('--no_resolve_msa_overlaps', action='store_true')
    ap.add_argument('--out_json', type=pathlib.Path, default=None)
    ap.add_argument('--out_csv', type=pathlib.Path, default=None)
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()

    # Ensure pipeline logs per-step timings.
    os.environ.setdefault('AF3_FEAT_TIMING', '1')
    # Make INFO-level absl logs visible on stdout/stderr.
    try:
        absl_logging.set_verbosity(absl_logging.INFO)
    except Exception:
        pass

    base = _load_fold_input(args.json)
    if args.peptide_chain_id and args.peptide:
        fold_in = _override_peptide_chain(base, args.peptide_chain_id, args.peptide)
        print(f"Overrode chain {args.peptide_chain_id} to sequence of length {len(args.peptide)}")
    else:
        fold_in = base

    ccd = chemical_components.cached_ccd(user_ccd=fold_in.user_ccd)
    buckets = _parse_buckets(args.buckets)
    resolve_msa_overlaps = not args.no_resolve_msa_overlaps

    def _one_run() -> float:
        t0 = time.perf_counter()
        _ = featurisation.featurise_input(
            fold_input=fold_in,
            ccd=ccd,
            buckets=buckets,
            resolve_msa_overlaps=resolve_msa_overlaps,
            verbose=args.verbose,
        )
        t1 = time.perf_counter()
        return t1 - t0

    # Warmup
    if args.warmup:
        print(f'Warmup x{args.warmup}...')
        for _ in range(args.warmup):
            _one_run()

    totals: list[float] = []
    print('Running featurisation benchmark...')
    for i in range(args.runs):
        dt = _one_run()
        totals.append(dt)
        print(f'Run {i+1}: total={dt:.2f}s (see logs for per-step breakdown)')

    avg = float(np.mean(totals)) if totals else 0.0
    print(f'\nAverage total time over {len(totals)} run(s): {avg:.2f}s')

    if args.out_json:
        with open(args.out_json, 'w') as f:
            json.dump({'totals': totals, 'avg_total': avg}, f, indent=2)
        print(f'Wrote JSON timings to {args.out_json}')

    if args.out_csv:
        with open(args.out_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['run', 'total_s'])
            for i, t in enumerate(totals, start=1):
                writer.writerow([i, t])
            writer.writerow(['avg', avg])
        print(f'Wrote CSV timings to {args.out_csv}')


if __name__ == '__main__':
    main()
