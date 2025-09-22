#!/usr/bin/env python3
"""
Compare AF3 featurised batches between:
- Original featurisation (as in run_alphafold.py / featurisation.featurise_input)
- Peptide-variant partial update (peptide_variant_screen.update_batch_for_new_peptide)

Usage example:
  python mytests/compare_peptide_features.py \
    --json /home/hehuang/Projects/NK2R/af3/test_peptide_variant_screen/test_template_dimer/all/fold_input.json \
    --peptide IKPGSFVPLF \
    [--peptide_chain_id C]

Notes:
- Detects the peptide chain by matching sequence length if --peptide_chain_id is not given.
- Assumes peptide custom MSA is a single-sequence A3M (the query only).
- Prints per-key shape/dtype and max abs diff summary; dumps a short discrepancy report.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import sys
from typing import Any, Dict, Iterable, Tuple

import numpy as np

# Allow running from repo root without installing.
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, (REPO_ROOT / 'src').as_posix())

from alphafold3.common import folding_input
from alphafold3.constants import chemical_components
from alphafold3.constants import mmcif_names
from alphafold3.data import featurisation
from alphafold3.data import msa as msa_module
from alphafold3.model import data3
from alphafold3.model import features


def _load_fold_input(json_path: pathlib.Path) -> folding_input.Input:
    with open(json_path, 'rt') as f:
        return folding_input.Input.from_json(f.read(), json_path)


def _sequence_to_aatype(sequence: str) -> np.ndarray:
    from alphafold3.constants import residue_names

    table = (
        residue_names.PROTEIN_TYPES_ONE_LETTER_WITH_UNKNOWN_AND_GAP_TO_INT
    )
    return np.asarray([table.get(a, table['X']) for a in sequence], dtype=np.int32)


def _chain_id_to_token_range(inp: folding_input.Input) -> Dict[str, Tuple[int, int]]:
    """Infer per-chain token start/end by chaining lengths of polymer chains.

    Assumes one token per standard polymer residue, matching AF3 tokeniser for
    standard residues.
    """
    mapping: Dict[str, Tuple[int, int]] = {}
    cur = 0
    for ch in inp.chains:
        # Only polymer chains contribute sequence tokens deterministically.
        # Ligands become single-atom tokens, but here we only need peptide positions.
        length = len(ch) if hasattr(ch, '__len__') else 0
        mapping[ch.id] = (cur, cur + length)
        cur += length
    return mapping


def _make_query_only_a3m(seq: str) -> str:
    return f">query\n{seq}\n"


def _build_original_features(
    base_input: folding_input.Input,
    peptide_chain_id: str,
    peptide_seq: str,
    *,
    buckets: Iterable[int] | None = None,
    resolve_msa_overlaps: bool = True,
) -> features.BatchDict:
    """Build features by replacing peptide chain sequence + custom MSA, then featurising."""
    # Construct a modified fold_input with peptide chain overridden.
    new_chains = []
    for ch in base_input.chains:
        if ch.id == peptide_chain_id and isinstance(ch, folding_input.ProteinChain):
            # Keep templates as-is; inject custom single-seq A3M for both paired/unpaired.
            new_chains.append(
                folding_input.ProteinChain(
                    id=ch.id,
                    sequence=peptide_seq,
                    ptms=ch.ptms,
                    paired_msa=_make_query_only_a3m(peptide_seq),
                    unpaired_msa=_make_query_only_a3m(peptide_seq),
                    templates=ch.templates,
                )
            )
        else:
            new_chains.append(ch)

    mod_input = dataclasses.replace(base_input, chains=tuple(new_chains))

    # Featurise with the same seed (use first) like run_alphafold.
    # Load the default CCD (chemical component dictionary).
    ccd = chemical_components.cached_ccd()
    batches = featurisation.featurise_input(
        fold_input=mod_input,
        ccd=ccd,
        buckets=tuple(buckets) if buckets is not None else None,
        ref_max_modified_date=None,
        conformer_max_iterations=None,
        resolve_msa_overlaps=resolve_msa_overlaps,
        verbose=False,
    )
    if not batches:
        raise RuntimeError('Featurisation produced no batches')
    return batches[0]


def _build_partial_update_features(
    base_batch: features.BatchDict,
    peptide_token_positions: np.ndarray,
    peptide_seq: str,
) -> features.BatchDict:
    """Apply the same partial update as peptide_variant_screen.update_batch_for_new_peptide."""
    # Deep-copy to avoid mutating shared state.
    out: features.BatchDict = {}
    for k, v in base_batch.items():
        if isinstance(v, np.ndarray):
            out[k] = v.copy()
        else:
            # Immutable or small scalars/strings are fine to reuse; if dicts appear, clone shallowly.
            try:
                import copy as _copy
                out[k] = _copy.deepcopy(v)
            except Exception:
                out[k] = v

    # Update aatype
    new_aatype = _sequence_to_aatype(peptide_seq)
    if 'aatype' not in out:
        raise KeyError('aatype missing in base batch')
    for i, pos in enumerate(peptide_token_positions):
        if i >= len(new_aatype):
            break
        out['aatype'][pos] = new_aatype[i]

    # Update MSA rows where peptide columns are non-gap
    try:
        from alphafold3.model import data_constants

        pep_cols = peptide_token_positions
        msa_arr = out['msa']  # (msa_rows, num_tokens)
        gap_idx = data_constants.MSA_GAP_IDX
        if pep_cols.size > 0:
            peptide_row_mask = (msa_arr[:, pep_cols] != gap_idx).any(axis=1)
            row_idx = np.where(peptide_row_mask)[0].astype(np.int32)
        else:
            row_idx = np.array([], dtype=np.int32)

        if pep_cols.size > 0 and row_idx.size > 0:
            # Encode the query-only sequence into Msa features
            a3m = _make_query_only_a3m(peptide_seq)
            pep_msa = msa_module.Msa.from_a3m(
                query_sequence=peptide_seq,
                chain_poly_type=mmcif_names.PROTEIN_CHAIN,
                a3m=a3m,
                deduplicate=False,
            )
            pep_feats = pep_msa.featurize()
            pep_msa_row = pep_feats['msa'][0]
            pep_del_row = pep_feats['deletion_matrix'][0]
            new_msa_block = np.tile(pep_msa_row, (row_idx.size, 1))
            new_del_block = np.tile(pep_del_row, (row_idx.size, 1))

            out['msa'][np.ix_(row_idx, pep_cols)] = new_msa_block.astype(out['msa'].dtype)
            out['deletion_matrix'][np.ix_(row_idx, pep_cols)] = new_del_block.astype(out['deletion_matrix'].dtype)

            prof = data3.get_profile_features(new_msa_block, new_del_block)
            out['profile'][pep_cols, :] = prof['profile'].astype(out['profile'].dtype)
            out['deletion_mean'][pep_cols] = prof['deletion_mean'].astype(out['deletion_mean'].dtype)
    except Exception as e:
        print(f"WARNING: MSA/profile update failed: {e}")

    return out


def _compare_batches(a: features.BatchDict, b: features.BatchDict, *, atol: float = 0.0, rtol: float = 0.0) -> Dict[str, Any]:
    keys_a = set(a.keys())
    keys_b = set(b.keys())
    only_a = sorted(list(keys_a - keys_b))
    only_b = sorted(list(keys_b - keys_a))
    common = sorted(list(keys_a & keys_b))

    diffs: Dict[str, Any] = {
        'only_in_partial': only_a,
        'only_in_original': only_b,
        'mismatch': {},  # key -> summary
        'match': [],
    }

    for k in common:
        va = a[k]
        vb = b[k]
        if isinstance(va, np.ndarray) and isinstance(vb, np.ndarray):
            same_shape = va.shape == vb.shape
            same_dtype = va.dtype == vb.dtype
            if not same_shape or not same_dtype:
                diffs['mismatch'][k] = {
                    'shape_partial': list(va.shape),
                    'shape_original': list(vb.shape),
                    'dtype_partial': str(va.dtype),
                    'dtype_original': str(vb.dtype),
                }
                continue
            # Numeric comparison
            if np.issubdtype(va.dtype, np.floating):
                delta = np.abs(va.astype(np.float64) - vb.astype(np.float64))
                max_abs = float(delta.max(initial=0.0))
                num_diff = int(np.count_nonzero(delta > (atol + rtol * np.abs(vb))))
                if num_diff == 0:
                    diffs['match'].append(k)
                else:
                    diffs['mismatch'][k] = {
                        'max_abs_diff': max_abs,
                        'num_diff': num_diff,
                        'numel': int(va.size),
                    }
            else:
                equal = (va == vb)
                num_diff = int((~equal).sum())
                if num_diff == 0:
                    diffs['match'].append(k)
                else:
                    diffs['mismatch'][k] = {
                        'num_diff': num_diff,
                        'numel': int(va.size),
                    }
        else:
            # Non-array types: compare directly if simple scalars
            same = va == vb
            if same:
                diffs['match'].append(k)
            else:
                diffs['mismatch'][k] = {
                    'partial': str(type(va)),
                    'original': str(type(vb)),
                }

    return diffs


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--json', type=pathlib.Path, required=True, help='Path to fold_input.json')
    p.add_argument('--peptide', type=str, required=True, help='Peptide sequence')
    p.add_argument('--peptide_chain_id', type=str, default=None, help='Chain id of peptide (optional)')
    p.add_argument('--dump_json', type=pathlib.Path, default=None, help='Optional path to dump comparison JSON')
    args = p.parse_args()

    base_input = _load_fold_input(args.json)

    # Identify peptide chain id if not provided.
    pep_len = len(args.peptide)
    chain_map = _chain_id_to_token_range(base_input)
    peptide_chain_id = args.peptide_chain_id
    if peptide_chain_id is None:
        candidates = [ch.id for ch in base_input.protein_chains if len(ch) == pep_len]
        if 'C' in candidates:
            peptide_chain_id = 'C'
        elif candidates:
            peptide_chain_id = candidates[0]
        else:
            raise ValueError('Could not infer peptide chain id. Please pass --peptide_chain_id.')

    print(f"Peptide chain id: {peptide_chain_id}")

    # Original features for the peptide sequence
    orig_batch = _build_original_features(
        base_input=base_input,
        peptide_chain_id=peptide_chain_id,
        peptide_seq=args.peptide,
        buckets=None,
        resolve_msa_overlaps=True,
    )

    # Base batch (from original base_input without peptide change)
    ccd = chemical_components.cached_ccd()
    base_batches = featurisation.featurise_input(
        fold_input=base_input,
        ccd=ccd,
        buckets=None,
        ref_max_modified_date=None,
        conformer_max_iterations=None,
        resolve_msa_overlaps=True,
        verbose=False,
    )
    if not base_batches:
        raise RuntimeError('Featurisation produced no base batches')
    base_batch = base_batches[0]

    # Compute peptide token positions from chain order and lengths.
    start, end = chain_map[peptide_chain_id]
    pep_positions = np.arange(start, end, dtype=np.int32)

    # Partial update on base batch
    part_batch = _build_partial_update_features(
        base_batch=base_batch,
        peptide_token_positions=pep_positions,
        peptide_seq=args.peptide,
    )

    # Compare feature dicts
    diffs = _compare_batches(part_batch, orig_batch)

    # Pretty print summary
    print('\n=== Feature Comparison Summary ===')
    if diffs['only_in_partial']:
        print('Only in partial-update batch:', ', '.join(diffs['only_in_partial']))
    if diffs['only_in_original']:
        print('Only in original batch:', ', '.join(diffs['only_in_original']))

    mismatches = diffs['mismatch']
    if mismatches:
        print('\nMismatched keys (shape/dtype or values):')
        for k, v in mismatches.items():
            print(f"- {k}: {json.dumps(v)}")
    else:
        print('\nAll common keys match exactly.')

    print(f"\nMatched keys: {len(diffs['match'])}, Mismatched: {len(mismatches)}")

    if args.dump_json:
        with open(args.dump_json, 'w') as f:
            json.dump(diffs, f, indent=2)
        print(f"\nWrote detailed diff to {args.dump_json}")


if __name__ == '__main__':
    main()
