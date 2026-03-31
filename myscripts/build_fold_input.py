#!/usr/bin/env python3
"""Generate a local AlphaFold 3 fold_input.json from AF3 server results."""

from __future__ import annotations

import argparse
import json
import re
import string
import sys
from pathlib import Path
from typing import Dict, List


class FoldInputBuilderError(RuntimeError):
    """Raised when the fold input cannot be built."""


def _load_job_request(result_dir: Path) -> Dict:
    job_files = sorted(result_dir.glob("*job_request.json"))
    if not job_files:
        raise FoldInputBuilderError(
            f"No *job_request.json file found under {result_dir}."
        )
    job_data = json.loads(job_files[0].read_text())
    if isinstance(job_data, list):
        if not job_data:
            raise FoldInputBuilderError(
                f"Job request file {job_files[0]} is empty."
            )
        job_data = job_data[0]
    if not isinstance(job_data, dict):
        raise FoldInputBuilderError(
            f"Job request {job_files[0]} has unexpected structure."
        )
    return job_data


def _build_chain_entries(job_data: Dict) -> List[Dict]:
    sequences = job_data.get("sequences")
    if not sequences:
        raise FoldInputBuilderError("Job request does not contain sequences.")
    chain_entries: List[Dict] = []
    for idx, seq_entry in enumerate(sequences):
        protein_entry = (
            seq_entry.get("proteinChain")
            or seq_entry.get("protein")
            or seq_entry
        )
        sequence = protein_entry.get("sequence")
        if not sequence:
            raise FoldInputBuilderError(
                f"Sequence entry #{idx} is missing the primary sequence."
            )
        chain_id = string.ascii_uppercase[idx]
        modifications = protein_entry.get("modifications") or []
        chain_entries.append(
            {
                "protein": {
                    "id": chain_id,
                    "sequence": sequence,
                    "modifications": modifications,
                    "templates": [],
                    "pairedMsaPath": "",
                    "unpairedMsaPath": "",
                }
            }
        )
    return chain_entries


def _attach_msas(result_dir: Path, chain_lookup: Dict[str, Dict]) -> None:
    msa_dir = result_dir / "msas"
    if not msa_dir.is_dir():
        return
    msa_re = re.compile(r"(paired|unpaired)_msa_chains_([a-z]+)\.")
    for msa_path in msa_dir.iterdir():
        if not msa_path.is_file():
            continue
        match = msa_re.search(msa_path.name)
        if not match:
            continue
        msa_kind, chain_token = match.groups()
        chain = chain_lookup.get(chain_token)
        if chain is None:
            print(
                f"Warning: could not map MSA {msa_path.name} to a chain.",
                file=sys.stderr,
            )
            continue
        resolved = str(msa_path.resolve())
        if msa_kind == "paired":
            chain["pairedMsaPath"] = resolved
        else:
            chain["unpairedMsaPath"] = resolved


def _attach_templates(result_dir: Path, chain_lookup: Dict[str, Dict]) -> None:
    templates_dir = result_dir / "templates"
    if not templates_dir.is_dir():
        return
    hits_pattern = re.compile(r"template_hits_chains_([a-z]+)_query_to_hit\.json$")
    for hits_file in templates_dir.glob("*_query_to_hit.json"):
        match = hits_pattern.search(hits_file.name)
        if not match:
            continue
        chain_token = match.group(1)
        chain = chain_lookup.get(chain_token)
        if chain is None:
            print(
                f"Warning: could not map template hit file {hits_file.name} "
                "to a chain.",
                file=sys.stderr,
            )
            continue
        hits_data = json.loads(hits_file.read_text())
        if not isinstance(hits_data, list):
            print(
                f"Warning: template hit file {hits_file} is malformed.",
                file=sys.stderr,
            )
            continue
        for hit in hits_data:
            name = hit.get("name")
            if not name:
                continue
            cif_path = templates_dir / name
            if not cif_path.exists():
                print(
                    f"Warning: template CIF {cif_path} referenced in "
                    f"{hits_file} does not exist.",
                    file=sys.stderr,
                )
                continue
            template_entry = {
                "mmcifPath": str(cif_path.resolve()),
                "queryIndices": hit.get("queryIndices", []),
                "templateIndices": hit.get("templateIndices", []),
            }
            chain["templates"].append(template_entry)


def _build_output_structure(result_dir: Path, job_data: Dict) -> Dict:
    chain_entries = _build_chain_entries(job_data)
    chain_lookup = {
        string.ascii_lowercase[i]: entry["protein"]
        for i, entry in enumerate(chain_entries)
    }
    _attach_msas(result_dir, chain_lookup)
    _attach_templates(result_dir, chain_lookup)

    dialect = job_data.get("dialect", "alphafold3")
    # Local folds typically expect the alphafold3 dialect.
    if dialect == "alphafoldserver":
        dialect = "alphafold3"

    model_seeds = job_data.get("modelSeeds") or []
    seeds: List[int] = []
    for seed in model_seeds:
        try:
            seeds.append(int(seed))
        except (TypeError, ValueError):
            print(f"Warning: skipping model seed {seed!r}", file=sys.stderr)

    return {
        "name": job_data.get("name", result_dir.name),
        "sequences": chain_entries,
        "modelSeeds": seeds,
        "dialect": dialect,
        "version": job_data.get("version", 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Construct a fold_input.json for local AlphaFold 3 runs using "
            "an extracted AF3 result directory."
        )
    )
    parser.add_argument(
        "result_dir", type=Path, help="Path to the AF3 result directory."
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Optional explicit path for the generated fold_input.json.",
    )
    args = parser.parse_args()

    result_dir = args.result_dir.expanduser().resolve()
    if not result_dir.is_dir():
        raise FoldInputBuilderError(f"{result_dir} is not a directory.")

    job_data = _load_job_request(result_dir)
    fold_input = _build_output_structure(result_dir, job_data)
    output_path = (
        args.output.expanduser().resolve()
        if args.output
        else Path.cwd() / "fold_input.json"
    )
    output_path.write_text(json.dumps(fold_input, indent=2))
    print(f"Wrote fold input to {output_path}")


if __name__ == "__main__":
    try:
        main()
    except FoldInputBuilderError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
