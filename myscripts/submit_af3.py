#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import subprocess
import sys


def read_fasta(path: Path):
    header = None
    seq_lines = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is None:
                    header = line[1:].split()[0]
                else:
                    raise ValueError("FASTA contains more than one sequence.")
                continue
            if header is None:
                raise ValueError("FASTA missing header line.")
            seq_lines.append(line.replace(" ", ""))
    if header is None or not seq_lines:
        raise ValueError("FASTA missing sequence.")
    return header, "".join(seq_lines)


def read_smi(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            smiles = parts[0]
            name = parts[1] if len(parts) > 1 else path.stem
            return name, smiles
    raise ValueError("SMI file missing SMILES line.")


def run_msa(fasta_path: Path, msa_dir: Path):
    msa_dir.mkdir(parents=True, exist_ok=True)
    cmd = "\n".join(
        [
            "source /home/hehuang/miniconda3/condaenv.sh",
            "conda activate colabfold",
            f"colabfold_batch {fasta_path} {msa_dir} --msa-only",
        ]
    )
    subprocess.run(["bash", "-lc", cmd], check=True)


def build_fold_input(
    name: str,
    sequence: str,
    smiles: str,
    use_msa: bool,
    msa_path: Path,
    num_seeds: int,
):
    protein_entry = {
        "id": "A",
        "sequence": sequence,
    }
    if use_msa:
        protein_entry.update(
            {
                "templates": [],
                "pairedMsa": "",
                "unpairedMsaPath": str(msa_path),
            }
        )
    return {
        "name": name,
        "sequences": [
            {"protein": protein_entry},
            {"ligand": {"id": "C", "smiles": smiles}},
        ],
        "modelSeeds": list(range(num_seeds)),
        "dialect": "alphafold3",
        "version": 1,
    }


def render_sbatch(job_name: str):
    return "\n".join(
        [
            "#!/bin/bash",
            f"#SBATCH --job-name={job_name}",
            "#SBATCH --output=run.out",
            "#SBATCH --error=run.out",
            "#SBATCH --nodes=1",
            "#SBATCH --ntasks=1",
            "#SBATCH --cpus-per-task=10",
            "#SBATCH --partition=oxygen",
            "#SBATCH --gres=gpu:1",
            "",
            "# Activate conda environment",
            "source /home/hehuang/miniconda3/etc/profile.d/conda.sh",
            "conda activate af3",
            "",
            "# Source and run the alphafold script",
            "af3_dir=/home/hehuang/Tools/alphafold3",
            "bash $af3_dir/run_alphafold.sh",
            "",
        ]
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create AF3 fold_input.json and submit a SLURM job for a single-chain "
            "receptor protein with one ligand."
        )
    )
    parser.add_argument(
        "rec_fasta",
        type=Path,
        help="Receptor FASTA with exactly one sequence (single-chain only).",
    )
    parser.add_argument(
        "lig_smi",
        type=Path,
        help="Ligand .smi file (first non-comment line used).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Output directory for fold_input.json and sbatch script.",
    )
    parser.add_argument(
        "--job-name",
        default="af3",
        help="SLURM job name for the generated sbatch script.",
    )
    parser.add_argument(
        "--mmseq2",
        action="store_true",
        help="Generate MSA with colabfold (mmseq2) and write unpairedMsaPath.",
    )
    parser.add_argument(
        "--msa-dir",
        type=Path,
        default=Path("msa"),
        help="Directory to place/read MSA outputs (used with --mmseq2).",
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=5,
        help="Number of model seeds; uses range(n) as modelSeeds.",
    )
    parser.add_argument(
        "--sbatch-out",
        type=Path,
        default=Path("run_af3.sbatch"),
        help="Output path for the generated sbatch script.",
    )
    args = parser.parse_args()

    rec_header, sequence = read_fasta(args.rec_fasta)
    lig_name, smiles = read_smi(args.lig_smi)
    fold_name = f"{rec_header}_{lig_name}"

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    msa_dir = args.msa_dir
    if not msa_dir.is_absolute():
        msa_dir = output_dir / msa_dir

    if args.mmseq2:
        run_msa(args.rec_fasta.resolve(), msa_dir)
        msa_path = msa_dir / f"{rec_header}.a3m"
    else:
        msa_path = msa_dir / f"{rec_header}.a3m"

    if args.num_seeds < 1:
        raise ValueError("--num-seeds must be >= 1.")

    fold_input = build_fold_input(
        fold_name, sequence, smiles, args.mmseq2, msa_path, args.num_seeds
    )
    fold_input_path = output_dir / "fold_input.json"
    with fold_input_path.open("w", encoding="utf-8") as handle:
        json.dump(fold_input, handle, indent=2)
        handle.write("\n")

    sbatch_contents = render_sbatch(args.job_name)

    sbatch_out_path = args.sbatch_out
    if not sbatch_out_path.is_absolute():
        sbatch_out_path = output_dir / sbatch_out_path

    sbatch_out_path.write_text(sbatch_contents, encoding="utf-8")

    subprocess.run(["sbatch", str(sbatch_out_path)], cwd=output_dir, check=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
