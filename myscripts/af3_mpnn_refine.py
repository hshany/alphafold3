#!/usr/bin/env python3
"""
Iterative AF3 + ProteinMPNN refinement.

For each iteration:
1. Build an AlphaFold3 model for the current peptide sequence using the
   AF3MCMCEvaluator (same evaluator used by af3_mcmc_optimize.py).
2. Convert the resulting CIF to PDB and launch ProteinMPNN via a generated bash
   script so it can activate its own environment. The redesigned peptide chain
   becomes the sequence for the next AF3 iteration.
"""

from __future__ import annotations

import argparse
import csv
import logging
import pathlib
import shlex
import subprocess
import sys
from dataclasses import dataclass
from typing import Optional

import gemmi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Iteratively alternate AF3 prediction and ProteinMPNN redesign."
    )
    parser.add_argument("--json_path", type=pathlib.Path, required=True)
    parser.add_argument("--model_dir", type=pathlib.Path, required=True)
    parser.add_argument("--peptide_chain_id", type=str, required=True)
    parser.add_argument("--receptor_chain_id", type=str, required=True)
    parser.add_argument("--steps", type=int, required=True,
                        help="Number of AF3→MPNN refinement iterations.")
    parser.add_argument("--output_dir", type=pathlib.Path, required=True)
    parser.add_argument("--myscripts_dir", type=pathlib.Path,
                        default=pathlib.Path("/home/hehuang/Tools/alphafold3/myscripts"))
    parser.add_argument("--num_diffusion_samples", type=int, default=1)
    parser.add_argument("--num_recycles", type=int, default=1)
    parser.add_argument("--loss_mode", type=str, choices=["neg_ranking", "composite"], default="composite")
    parser.add_argument("--w_rank", type=float, default=0.0)
    parser.add_argument("--w_plddt", type=float, default=0.0)
    parser.add_argument("--w_iptm", type=float, default=1.0)
    parser.add_argument("--w_pae", type=float, default=0.0)
    parser.add_argument("--mutation_bias", type=str, choices=["uniform", "peptide_plddt"], default="uniform")
    parser.add_argument("--initial_sequence", type=str, default=None,
                        help="Optional custom starting peptide sequence; defaults to sequence in json.")
    parser.add_argument("--proteinmpnn_root", type=pathlib.Path,
                        default=pathlib.Path("/home/hehuang/Tools/ProteinMPNN"))
    parser.add_argument("--mpnn_env_cmd", type=str,
                        default="source /home/hehuang/miniconda3/condaenv.sh && conda activate RFAA",
                        help="Shell snippet run inside each generated MPNN script before launching ProteinMPNN.")
    parser.add_argument("--mpnn_design_chains", type=str, default=None,
                        help="Chain IDs (space separated) passed to ProteinMPNN assign_fixed_chains. "
                        "Defaults to --peptide_chain_id.")
    parser.add_argument("--fixed_positions", type=str, default=None,
                        help="Comma/space separated 1-based residue indices to keep fixed during ProteinMPNN. "
                        "Negative indices count from the peptide C-terminus (e.g., -1 = last residue).")
    parser.add_argument("--mpnn_num_seq_per_target", type=int, default=1)
    parser.add_argument("--mpnn_sampling_temp", type=str, default="0.1")
    parser.add_argument("--mpnn_seed", type=int, default=37)
    parser.add_argument("--mpnn_batch_size", type=int, default=1)
    parser.add_argument("--mpnn_additional_args", type=str, default="--use_soluble_model",
                        help="Additional args appended to protein_mpnn_run.py invocation.")
    parser.add_argument("--summary_csv", type=pathlib.Path, default=None,
                        help="Optional custom path for the summary CSV; defaults to output_dir/iteration_summary.csv")
    return parser.parse_args()


@dataclass
class MPNNScriptConfig:
    proteinmpnn_root: pathlib.Path
    folder_with_pdbs: pathlib.Path
    output_dir: pathlib.Path
    chains_to_design: str
    design_positions: str
    num_seq_per_target: int
    sampling_temp: str
    seed: int
    batch_size: int
    additional_args: str
    env_cmd: str


def _quote(val: pathlib.Path | str) -> str:
    return shlex.quote(str(val))


def _parse_fixed_positions(spec: Optional[str], seq_len: int) -> set[int]:
    if not spec:
        return set()
    tokens = spec.replace(",", " ").split()
    fixed: set[int] = set()
    for tok in tokens:
        tok = tok.strip()
        if not tok:
            continue
        try:
            idx = int(tok)
        except ValueError as exc:
            raise ValueError(f"Invalid fixed position '{tok}'") from exc
        if idx == 0:
            raise ValueError("Position index 0 is invalid; use 1-based or negative-from-end indices.")
        if idx < 0:
            pos = seq_len + idx + 1  # e.g., -1 => seq_len
        else:
            pos = idx
        if pos < 1 or pos > seq_len:
            raise ValueError(f"Fixed position {tok} resolves to {pos}, outside peptide length {seq_len}")
        fixed.add(pos)
    return fixed


def _designable_positions_string(seq_len: int, fixed_spec: Optional[str]) -> str:
    fixed = _parse_fixed_positions(fixed_spec, seq_len)
    designable = [str(i) for i in range(1, seq_len + 1) if i not in fixed]
    if not designable:
        raise ValueError("All peptide positions are fixed; nothing left for ProteinMPNN to design.")
    return " ".join(designable)


def ensure_pdb_outputs(structure_dir: pathlib.Path) -> None:
    cif_files = sorted(structure_dir.glob("*.cif"))
    if not cif_files:
        raise FileNotFoundError(f"No CIF files found under {structure_dir}")
    for cif_path in cif_files:
        pdb_path = cif_path.with_suffix(".pdb")
        if pdb_path.exists():
            continue
        st = gemmi.read_structure(str(cif_path))
        st.write_minimal_pdb(str(pdb_path))


def render_mpnn_script(script_path: pathlib.Path, cfg: MPNNScriptConfig) -> None:
    script_lines = [
        "#!/bin/bash",
        "set -euo pipefail",
        "",
    ]
    env_cmd = cfg.env_cmd.strip()
    if env_cmd:
        script_lines.append(env_cmd)
        script_lines.append("")

    folder_with_pdbs = _quote(cfg.folder_with_pdbs)
    output_dir = _quote(cfg.output_dir)
    parsed = _quote(cfg.output_dir / "parsed_pdbs.jsonl")
    assigned = _quote(cfg.output_dir / "assigned_pdbs.jsonl")
    fixed = _quote(cfg.output_dir / "fixed_pdbs.jsonl")
    chains = cfg.chains_to_design
    positions = cfg.design_positions
    script_lines.extend([
        f"protein_mpnn={_quote(cfg.proteinmpnn_root)}",
        f"folder_with_pdbs={folder_with_pdbs}",
        f"output_dir={output_dir}",
        f"path_for_parsed_chains={parsed}",
        f"path_for_assigned_chains={assigned}",
        f"path_for_fixed_positions={fixed}",
        f'chains_to_design="{chains}"',
        f'design_positions="{positions}"',
        "",
        'if [ ! -d "$output_dir" ]; then',
        '    mkdir -p "$output_dir"',
        "fi",
        "",
        'python "$protein_mpnn"/helper_scripts/parse_multiple_chains.py '
        '--input_path="$folder_with_pdbs" --output_path="$path_for_parsed_chains"',
        'python "$protein_mpnn"/helper_scripts/assign_fixed_chains.py '
        '--input_path="$path_for_parsed_chains" --output_path="$path_for_assigned_chains" '
        '--chain_list "$chains_to_design"',
        'python "$protein_mpnn"/helper_scripts/make_fixed_positions_dict.py '
        '--input_path="$path_for_parsed_chains" --output_path="$path_for_fixed_positions" '
        '--chain_list "$chains_to_design" --position_list "$design_positions" --specify_non_fixed',
        "",
        'python "$protein_mpnn"/protein_mpnn_run.py \\',
        '        --jsonl_path "$path_for_parsed_chains" \\',
        '        --chain_id_jsonl "$path_for_assigned_chains" \\',
        '        --fixed_positions_jsonl "$path_for_fixed_positions" \\',
        '        --out_folder "$output_dir" \\',
        f'        --num_seq_per_target {cfg.num_seq_per_target} \\',
        f'        --sampling_temp "{cfg.sampling_temp}" \\',
        f'        --seed {cfg.seed} \\',
        f'        --batch_size {cfg.batch_size} \\',
        f'        {cfg.additional_args}',
        "",
    ])
    script_path.write_text("\n".join(script_lines))
    script_path.chmod(0o750)


def parse_fasta_with_scores(fasta_path: pathlib.Path) -> list[dict]:
    entries: list[dict] = []
    header: Optional[str] = None
    seq_lines: list[str] = []
    with open(fasta_path, "r") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header and seq_lines:
                    entries.append({"header": header, "sequence": "".join(seq_lines)})
                header = line[1:].strip()
                seq_lines = []
            else:
                seq_lines.append(line)
        if header and seq_lines:
            entries.append({"header": header, "sequence": "".join(seq_lines)})

    for entry in entries:
        entry["score"] = _extract_score(entry["header"])
    return entries


def _extract_score(header: str) -> Optional[float]:
    import re

    match = re.search(r"\bscore=([\-0-9.]+)", header)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def pick_best_mpnn_sequence(seq_dir: pathlib.Path) -> Optional[dict]:
    if not seq_dir.is_dir():
        return None
    best: Optional[dict] = None
    for fasta_path in sorted(seq_dir.glob("*.fa")):
        entries = parse_fasta_with_scores(fasta_path)
        if not entries:
            continue
        # ProteinMPNN writes the original sequence first; skip it.
        for entry in entries[1:]:
            entry["source"] = str(fasta_path)
            if best is None:
                best = entry
                continue
            if entry["score"] is not None and best["score"] is None:
                best = entry
                continue
            if entry["score"] is not None and best["score"] is not None:
                if entry["score"] > best["score"]:
                    best = entry
                continue
    return best


def append_summary_row(csv_path: pathlib.Path, row: list) -> None:
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as handle:
        writer = csv.writer(handle)
        if write_header:
            writer.writerow([
                "iteration",
                "input_sequence",
                "af3_loss",
                "ranking_score",
                "plddt",
                "iptm",
                "pae",
                "mpnn_score",
                "output_sequence",
                "af3_dir",
                "mpnn_dir",
                "mpnn_fasta",
            ])
        writer.writerow(row)


def main() -> None:
    args = parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    log = logging.getLogger("af3-mpnn-refine")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = args.summary_csv or args.output_dir / "iteration_summary.csv"
    if not args.mpnn_design_chains:
        args.mpnn_design_chains = args.peptide_chain_id

    myscripts_dir = args.myscripts_dir.resolve()
    if str(myscripts_dir) not in sys.path:
        sys.path.insert(0, str(myscripts_dir))

    from af3_mcmc_optimize import AF3MCMCEvaluator, LossConfig  # type: ignore
    from peptide_variant_screen_1seed import write_variant_outputs  # type: ignore

    loss_cfg = LossConfig(
        mode=args.loss_mode,
        w_rank=args.w_rank,
        w_plddt=args.w_plddt,
        w_iptm=args.w_iptm,
        w_pae=args.w_pae,
    )
    log.info("Compiling AF3 model via evaluator (first call may take longer)...")
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
    )

    current_seq = args.initial_sequence or evaluator.initial_seq
    if not current_seq:
        raise SystemExit("Unable to determine initial peptide sequence.")
    log.info("Starting sequence (%d aa): %s", len(current_seq), current_seq)

    for iter_idx in range(args.steps):
        iter_label = f"iter_{iter_idx:03d}"
        iter_dir = args.output_dir / iter_label
        af3_dir = iter_dir / "af3"
        mpnn_dir = iter_dir / "mpnn"
        af3_dir.mkdir(parents=True, exist_ok=True)
        mpnn_dir.mkdir(parents=True, exist_ok=True)

        log.info("[%s] Running AF3 evaluation...", iter_label)
        eval_out = evaluator.evaluate(current_seq)
        metrics = eval_out["metrics"]
        loss_val = eval_out["loss"]
        write_variant_outputs(
            inference_results=eval_out["inference_results"],
            variant_dir=af3_dir,
            job_name=f"{iter_label}_af3",
            seed=int(eval_out.get("seed") or 0),
            write_individual_samples=False,
            tgz_confidences=True,
        )
        try:
            ensure_pdb_outputs(af3_dir)
        except Exception as exc:
            raise SystemExit(f"Failed to ensure PDB outputs for {iter_label}: {exc}") from exc
        log.info(
            "[%s] AF3 ranking=%.4f pLDDT=%.2f ipTM=%.4f loss=%.4f",
            iter_label,
            float(metrics.get("ranking_score", float("nan"))),
            float(metrics.get("plddt", float("nan"))),
            float(metrics.get("iptm", float("nan"))),
            float(loss_val),
        )

        design_positions = _designable_positions_string(len(current_seq), args.fixed_positions)
        mpnn_script = iter_dir / f"run_mpnn_{iter_label}.sh"
        mpnn_cfg = MPNNScriptConfig(
            proteinmpnn_root=args.proteinmpnn_root,
            folder_with_pdbs=af3_dir,
            output_dir=mpnn_dir,
            chains_to_design=args.mpnn_design_chains,
            design_positions=design_positions,
            num_seq_per_target=args.mpnn_num_seq_per_target,
            sampling_temp=args.mpnn_sampling_temp,
            seed=int(args.mpnn_seed) + int(iter_idx),
            batch_size=args.mpnn_batch_size,
            additional_args=args.mpnn_additional_args,
            env_cmd=args.mpnn_env_cmd,
        )
        render_mpnn_script(mpnn_script, mpnn_cfg)
        log.info("[%s] Launching ProteinMPNN via %s", iter_label, mpnn_script)
        try:
            subprocess.run(["bash", str(mpnn_script)], check=True)
        except subprocess.CalledProcessError as exc:
            raise SystemExit(f"ProteinMPNN run failed for {iter_label}: {exc}") from exc

        seqs_dir = mpnn_dir / "seqs"
        best_entry = pick_best_mpnn_sequence(seqs_dir)
        if not best_entry:
            raise SystemExit(f"No ProteinMPNN sequences found under {seqs_dir}")
        next_seq = best_entry["sequence"]
        mpnn_score = best_entry.get("score")
        log.info(
            "[%s] Selected ProteinMPNN sequence (score=%s): %s",
            iter_label,
            "n/a" if mpnn_score is None else f"{mpnn_score:.4f}",
            next_seq,
        )

        append_summary_row(
            summary_csv,
            [
                iter_idx,
                current_seq,
                float(loss_val),
                metrics.get("ranking_score"),
                metrics.get("plddt"),
                metrics.get("iptm"),
                metrics.get("pae"),
                mpnn_score,
                next_seq,
                str(af3_dir),
                str(mpnn_dir),
                best_entry.get("source"),
            ],
        )

        current_seq = next_seq

    log.info("Refinement complete. Summary: %s", summary_csv)


if __name__ == "__main__":
    main()
