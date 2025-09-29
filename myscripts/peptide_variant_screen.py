#!/usr/bin/env python3
"""
Optimized AlphaFold3 Peptide Variant Screening Script

This script implements an efficient workflow for screening multiple peptide sequences
against a fixed protein complex. It reuses JAX compilation and invariant features
to achieve ~6-7x speedup over naive approaches.

Usage:
    python peptide_variant_screen.py \
        --json_path complex.json \
        --peptide_chain_id C \
        --peptide_sequences peptides.fasta \
        --output_dir results/ \
        --model_dir path/to/model \
        --num_diffusion_samples 1 \
        --num_recycles 3
"""

import argparse
import copy
import json
import logging
import pathlib
import time
import sys
import traceback
import os
import datetime
from typing import Dict, List, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np

# AlphaFold3 imports
from alphafold3.common import folding_input
from alphafold3.constants import chemical_components
from alphafold3.constants import residue_names  
from alphafold3.constants import mmcif_names
from alphafold3.data import featurisation
from alphafold3.model import model
from alphafold3.model.pipeline import pipeline
from alphafold3.model import features
from alphafold3.data import msa as msa_module
from alphafold3.model import data3
from alphafold3.model import data_constants
from alphafold3.model import post_processing
import alphafold3.cpp
from alphafold3.constants import mmcif_names as _mmcif
from alphafold3.constants import chemical_components as _chem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('peptide_screening.log')
    ]
)
logger = logging.getLogger(__name__)


def load_fold_input(json_path: pathlib.Path) -> folding_input.Input:
    """Load fold input from JSON file."""
    logger.info(f"Loading fold input from {json_path}")
    with open(json_path, 'r') as f:
        json_str = f.read()
    return folding_input.Input.from_json(json_str, json_path)


def load_peptide_sequences(fasta_path: pathlib.Path) -> List[tuple[str, str]]:
    """Load peptide sequences from FASTA file."""
    logger.info(f"Loading peptide sequences from {fasta_path}")
    sequences = []
    
    with open(fasta_path, 'r') as f:
        current_name = None
        current_seq = ""
        
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_name is not None:
                    sequences.append((current_name, current_seq))
                current_name = line[1:]  # Remove '>'
                current_seq = ""
            else:
                current_seq += line
        
        # Add last sequence
        if current_name is not None:
            sequences.append((current_name, current_seq))
    
    logger.info(f"Loaded {len(sequences)} peptide sequences")
    return sequences


def sequence_to_aatype(sequence: str) -> List[int]:
    """Convert protein sequence to aatype integers."""
    return [
        residue_names.PROTEIN_TYPES_ONE_LETTER_WITH_UNKNOWN_AND_GAP_TO_INT.get(aa, 
            residue_names.PROTEIN_TYPES_ONE_LETTER_WITH_UNKNOWN_AND_GAP_TO_INT['X'])
        for aa in sequence
    ]


def find_chain_token_positions(batch: features.BatchDict, chain_id: str) -> np.ndarray:
    """Find token positions for a specific chain in the batch."""
    logger.debug(f"Finding token positions for chain {chain_id}")
    
    chain_positions = []
    
    # Method 1: Try to get chain info from convert_model_output
    try:
        if 'convert_model_output' in batch:
            convert_output = batch['convert_model_output']
            if hasattr(convert_output, 'all_token_atoms_layout'):
                all_token_atoms_layout = convert_output.all_token_atoms_layout
                
                # Find positions where chain_id matches
                for i, token_chain_id in enumerate(all_token_atoms_layout.chain_id):
                    if token_chain_id == chain_id:
                        chain_positions.append(i)
                
                if chain_positions:
                    positions = np.array(chain_positions, dtype=np.int32)
                    logger.info(f"Found {len(positions)} tokens for chain {chain_id}")
                    return positions
    except (AttributeError, KeyError) as e:
        logger.debug(f"Method 1 failed: {e}")
    
    # Method 2: Try to infer from batch structure
    # This is a heuristic approach for when detailed chain mapping isn't available
    try:
        # Look for chain information in token features or other batch components
        if 'token_features' in batch:
            token_features = batch['token_features']
            
            # If the batch preserves chain ordering and we know the chain structure,
            # we can make educated guesses about token positions
            # This requires knowledge of the input structure
            
            logger.warning(f"Using heuristic chain detection for chain {chain_id}")
            logger.warning("This may not be accurate - implement proper chain tracking")
            
            # Try to use seq_mask or other available info
            if 'seq_mask' in batch:
                seq_mask = batch['seq_mask']
                total_tokens = np.sum(seq_mask)
                logger.debug(f"Total tokens from seq_mask: {total_tokens}")
                
                # For chain C (typically the last/peptide chain), use heuristic
                if chain_id == 'C':
                    # Assume peptide is the last 10 positions
                    chain_positions = list(range(max(0, total_tokens - 10), total_tokens))
                    
                    if chain_positions:
                        positions = np.array(chain_positions, dtype=np.int32)
                        logger.warning(f"Using heuristic: assigned positions {positions} to chain {chain_id}")
                        return positions
    except (AttributeError, KeyError) as e:
        logger.debug(f"Method 2 failed: {e}")
    
    # If all methods fail, return empty array
    logger.error(f"Could not find token positions for chain {chain_id}")
    logger.error("You may need to implement custom chain mapping logic")
    logger.error("Consider tracking chain boundaries during featurization")
    return np.array([], dtype=np.int32)


def create_chain_mapping_from_fold_input(fold_input: folding_input.Input) -> Dict[str, tuple[int, int]]:
    """Create a mapping from chain IDs to token ranges.
    
    This function analyzes the fold_input to determine where each chain's 
    tokens will be located in the final batch.
    
    Returns:
        Dictionary mapping chain_id -> (start_token, end_token)
    """
    chain_mapping = {}
    current_token = 0
    
    logger.info("Creating chain mapping from fold_input")
    
    for chain in fold_input.chains:
        chain_id = chain.id
        
        if hasattr(chain, 'sequence'):
            sequence_length = len(chain.sequence)
        else:
            # For non-sequence chains (ligands), estimate token count
            sequence_length = 1
        
        start_token = current_token
        end_token = current_token + sequence_length
        
        chain_mapping[chain_id] = (start_token, end_token)
        current_token = end_token
        
        logger.debug(f"Chain {chain_id}: tokens {start_token}-{end_token} (length: {sequence_length})")
    
    return chain_mapping


def update_batch_for_new_peptide(
    base_batch: features.BatchDict, 
    new_sequence: str,
    peptide_positions: np.ndarray,
    chain_id: str,
    peptide_row_indices: Optional[np.ndarray] = None
) -> features.BatchDict:
    """Update batch features for a new peptide sequence."""
    logger.debug(f"Updating batch for peptide chain {chain_id} with sequence: {new_sequence}")
    
    # Create a copy of the base batch
    updated_batch = copy.deepcopy(base_batch)
    
    # Update aatype directly (it's a top-level key in the batch)
    new_aatype = sequence_to_aatype(new_sequence)
    
    if len(peptide_positions) > 0:
        # Update the aatype array at peptide positions
        current_aatype = updated_batch['aatype']
        for i, pos in enumerate(peptide_positions):
            if i < len(new_aatype) and pos < len(current_aatype):
                current_aatype[pos] = new_aatype[i]
        
        logger.debug(f"Updated positions {peptide_positions} with new sequence: {new_sequence}")
        logger.debug(f"New aatype values: {[current_aatype[pos] for pos in peptide_positions[:len(new_aatype)]]}")
    
    # Update MSA features for peptide chain, if possible
    try:
        if peptide_row_indices is None:
            peptide_row_indices = np.array([], dtype=np.int32)
        if len(peptide_positions) > 0 and len(peptide_row_indices) > 0:
            # Build query-only A3M to encode new sequence consistently
            a3m = f">query\n{new_sequence}\n"
            pep_msa = msa_module.Msa.from_a3m(
                query_sequence=new_sequence,
                chain_poly_type=mmcif_names.PROTEIN_CHAIN,
                a3m=a3m,
                deduplicate=False,
            )
            pep_feats = pep_msa.featurize()
            pep_msa_row = pep_feats['msa'][0]
            pep_del_row = pep_feats['deletion_matrix'][0]

            if len(pep_msa_row) != len(peptide_positions):
                logger.error(
                    "Peptide MSA row length (%d) does not match peptide token positions (%d)",
                    len(pep_msa_row), len(peptide_positions)
                )

            # Construct new peptide blocks and overwrite the full peptide columns deterministically:
            # 1) Set all MSA rows at peptide columns to GAP and deletions to 0
            # 2) Write the peptide query row(s) exactly at the cached row indices
            msa_dtype = updated_batch['msa'].dtype
            del_dtype = updated_batch['deletion_matrix'].dtype
            gap_idx = data_constants.MSA_GAP_IDX

            # Only fill the peptide row(s) with the encoded query sequence; do not
            # modify other rows at these columns to preserve original encoding.

            # Now fill the peptide row(s) with the encoded query sequence
            new_msa_block = np.tile(pep_msa_row, (len(peptide_row_indices), 1)).astype(msa_dtype)
            new_del_block = np.tile(pep_del_row, (len(peptide_row_indices), 1)).astype(del_dtype)

            updated_batch['msa'][np.ix_(peptide_row_indices, peptide_positions)] = new_msa_block
            updated_batch['deletion_matrix'][np.ix_(peptide_row_indices, peptide_positions)] = new_del_block

            # Recompute peptide column profile/deletion_mean from the new peptide MSA rows only
            prof = data3.get_profile_features(new_msa_block, new_del_block)
            updated_batch['profile'][peptide_positions, :] = prof['profile'].astype(updated_batch['profile'].dtype)
            updated_batch['deletion_mean'][peptide_positions] = prof['deletion_mean'].astype(updated_batch['deletion_mean'].dtype)

            logger.debug(
                f"Updated peptide MSA rows {peptide_row_indices.tolist()} and columns {peptide_positions.tolist()}"
            )
        else:
            logger.debug("Skipping MSA update: no peptide positions or peptide MSA rows detected.")
    except Exception as e:
        logger.warning(f"Failed to update MSA features for peptide chain {chain_id}: {e}")

    logger.debug(f"Updated aatype for {len(peptide_positions)} positions")
    return updated_batch


def _make_query_only_a3m(seq: str) -> str:
    return f">query\n{seq}\n"


def _fold_input_with_query_only_msas(
    base_input: folding_input.Input,
    *,
    peptide_chain_id: str,
    new_peptide_seq: str,
) -> folding_input.Input:
    import dataclasses as _dc
    new_chains = []
    for ch in base_input.chains:
        if isinstance(ch, folding_input.ProteinChain):
            if ch.id == peptide_chain_id:
                new_chains.append(
                    folding_input.ProteinChain(
                        id=ch.id,
                        sequence=new_peptide_seq,
                        ptms=ch.ptms,
                        paired_msa=_make_query_only_a3m(new_peptide_seq),
                        unpaired_msa=_make_query_only_a3m(new_peptide_seq),
                        templates=ch.templates,
                    )
                )
            else:
                new_chains.append(
                    folding_input.ProteinChain(
                        id=ch.id,
                        sequence=ch.sequence,
                        ptms=ch.ptms,
                        paired_msa=_make_query_only_a3m(ch.sequence),
                        unpaired_msa=_make_query_only_a3m(ch.sequence),
                        templates=ch.templates,
                    )
                )
        elif isinstance(ch, folding_input.RnaChain):
            new_chains.append(
                folding_input.RnaChain(
                    id=ch.id,
                    sequence=ch.sequence,
                    paired_msa=_make_query_only_a3m(ch.sequence),
                    unpaired_msa=_make_query_only_a3m(ch.sequence),
                    templates=ch.templates,
                )
            )
        else:
            new_chains.append(ch)
    return _dc.replace(base_input, chains=tuple(new_chains))

def _featurise_input(
    fold_input: folding_input.Input,
    ccd: chemical_components.Ccd,
    buckets: Sequence[int] | None,
    ref_max_modified_date: datetime.date | None = None,
    conformer_max_iterations: int | None = None,
    resolve_msa_overlaps: bool = True,
    verbose: bool = False,
) -> Sequence[features.BatchDict]:
  """Featurise the folding input.
  Copied from data/featurisation.py with deterministic_frames=False

  Args:
    fold_input: The input to featurise.
    ccd: The chemical components dictionary.
    buckets: Bucket sizes to pad the data to, to avoid excessive re-compilation
      of the model. If None, calculate the appropriate bucket size from the
      number of tokens. If not None, must be a sequence of at least one integer,
      in strictly increasing order. Will raise an error if the number of tokens
      is more than the largest bucket size.
    ref_max_modified_date: Optional maximum date that controls whether to allow
      use of model coordinates for a chemical component from the CCD if RDKit
      conformer generation fails and the component does not have ideal
      coordinates set. Only for components that have been released before this
      date the model coordinates can be used as a fallback.
    conformer_max_iterations: Optional override for maximum number of iterations
      to run for RDKit conformer search.
    resolve_msa_overlaps: Whether to deduplicate unpaired MSA against paired
      MSA. The default behaviour matches the method described in the AlphaFold 3
      paper. Set this to false if providing custom paired MSA using the unpaired
      MSA field to keep it exactly as is as deduplication against the paired MSA
      could break the manually crafted pairing between MSA sequences.
    verbose: Whether to print progress messages.

  Returns:
    A featurised batch for each rng_seed in the input.
  """
  featurisation.validate_fold_input(fold_input)

  # Set up data pipeline for single use.
  data_pipeline = pipeline.WholePdbPipeline(
      config=pipeline.WholePdbPipeline.Config(
          buckets=buckets,
          ref_max_modified_date=ref_max_modified_date,
          conformer_max_iterations=conformer_max_iterations,
          resolve_msa_overlaps=resolve_msa_overlaps,
          deterministic_frames=False
      ),
  )

  batches = []
  for rng_seed in fold_input.rng_seeds:
    featurisation_start_time = time.time()
    if verbose:
      print(f'Featurising data with seed {rng_seed}.')
    batch = data_pipeline.process_item(
        fold_input=fold_input,
        ccd=ccd,
        random_state=np.random.RandomState(rng_seed),
        random_seed=rng_seed,
    )
    if verbose:
      print(
          f'Featurising data with seed {rng_seed} took'
          f' {time.time() - featurisation_start_time:.2f} seconds.'
      )
    batches.append(batch)

  return batches


def _featurise_input_single(
    fold_input_inst: folding_input.Input,
) -> features.BatchDict:
    ccd = _chem.cached_ccd()
    batches = _featurise_input(
        fold_input=fold_input_inst,
        ccd=ccd,
        buckets=None,
        ref_max_modified_date=None,
        conformer_max_iterations=None,
        resolve_msa_overlaps=True,
        verbose=False,
    )
    if not batches:
        raise RuntimeError("Featurisation produced no batches")
    return batches[0]


def build_variant_features_with_msa_merge(
    *,
    base_batch: features.BatchDict,
    base_fold_input: folding_input.Input,
    peptide_chain_id: str,
    peptide_sequence: str,
    peptide_positions: np.ndarray,
    peptide_row_indices: Optional[np.ndarray],
) -> features.BatchDict:
    updated_input = _fold_input_with_query_only_msas(
        base_fold_input, peptide_chain_id=peptide_chain_id, new_peptide_seq=peptide_sequence
    )
    fresh_batch = _featurise_input_single(updated_input)

    updated_msa_batch = update_batch_for_new_peptide(
        base_batch=base_batch,
        new_sequence=peptide_sequence,
        peptide_positions=peptide_positions,
        chain_id=peptide_chain_id,
        peptide_row_indices=peptide_row_indices,
    )

    for k in (
        'msa',
        'msa_mask',
        'deletion_matrix',
        'profile',
        'deletion_mean',
        'num_alignments',
    ):
        if k in updated_msa_batch:
            fresh_batch[k] = updated_msa_batch[k]

    return fresh_batch
def _sanitize_name(name: str) -> str:
    """Sanitize a name for safe filesystem usage."""
    safe = []
    for ch in name:
        if ch.isalnum() or ch in ["_", "-", "."]:
            safe.append(ch)
        else:
            safe.append("_")
    # Avoid empty names
    out = "".join(safe).strip("._-")
    return out or "variant"


def write_variant_outputs(
    *,
    inference_results: Sequence[model.InferenceResult],
    variant_dir: pathlib.Path,
    job_name: str,
    seed: int,
    write_individual_samples: bool = True,
    tgz_confidences: bool = False,
) -> Dict:
    """Write processed outputs for a variant, mirroring run_alphafold.write_outputs.

    Returns a small summary dict with ranking scores and top selection.
    """
    variant_dir.mkdir(parents=True, exist_ok=True)

    # Terms of use text (same source as run_alphafold.py)
    output_terms = (
        pathlib.Path(alphafold3.cpp.__file__).parent / 'OUTPUT_TERMS_OF_USE.md'
    ).read_text()

    ranking_scores = []
    max_ranking_score = None
    max_ranking_result = None

    def _tgz_conf_json(out_dir: pathlib.Path, base_name: str) -> None:
        """Create a .tgz archive of the confidences JSON and remove the JSON."""
        try:
            import tarfile
            json_path = out_dir / f"{base_name}_confidences.json"
            if not json_path.exists():
                return
            tgz_path = out_dir / f"{base_name}_confidences.json.tgz"
            with tarfile.open(tgz_path, mode="w:gz") as tf:
                tf.add(json_path, arcname=f"{base_name}_confidences.json")
            try:
                os.remove(json_path)
            except OSError as e:
                logger.warning(f"Failed to remove original confidences.json: {e}")
        except Exception as e:
            logger.warning(f"Failed to tgz confidences JSON: {e}")

    for sample_idx, result in enumerate(inference_results):
        if write_individual_samples:
            sample_dir = variant_dir / f'seed-{seed}_sample-{sample_idx}'
            sample_dir.mkdir(parents=True, exist_ok=True)
            sample_name = f'{job_name}_seed-{seed}_sample-{sample_idx}'
            post_processing.write_output(
                inference_result=result,
                output_dir=sample_dir.as_posix(),
                name=sample_name,
            )
            if tgz_confidences:
                _tgz_conf_json(sample_dir, sample_name)
        rs = float(result.metadata['ranking_score'])
        ranking_scores.append((seed, sample_idx, rs))
        if max_ranking_score is None or rs > max_ranking_score:
            max_ranking_score = rs
            max_ranking_result = result

    # Write top-ranked output in the variant root and a ranking CSV
    if max_ranking_result is not None:
        post_processing.write_output(
            inference_result=max_ranking_result,
            output_dir=variant_dir.as_posix(),
            terms_of_use=output_terms,
            name=job_name,
        )
        if tgz_confidences:
            _tgz_conf_json(variant_dir, job_name)
        import csv
        with open(variant_dir / f'{job_name}_ranking_scores.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['seed', 'sample', 'ranking_score'])
            writer.writerows(ranking_scores)

    return {
        'job_name': job_name,
        'seed': seed,
        'ranking_scores': ranking_scores,
        'top_ranking_score': max_ranking_score,
    }


class OptimizedPeptideRunner:
    """Optimized runner for peptide variant screening."""
    
    def __init__(
        self, 
        model_dir: pathlib.Path, 
        device: Optional[jax.Device] = None,
        num_diffusion_samples: int = 1,
        num_recycles: int = 3
    ):
        """Initialize the optimized runner."""
        self.model_dir = model_dir
        self.device = device or jax.devices()[0]
        self.num_diffusion_samples = num_diffusion_samples
        self.num_recycles = num_recycles
        
        # Cached components
        self.model_runner = None
        self.base_batch = None
        self.peptide_positions = None
        self.peptide_row_indices = None
        self.is_compiled = False
        
        logger.info(f"Initialized OptimizedPeptideRunner on device: {self.device}")
        logger.info(f"Model config: num_diffusion_samples={num_diffusion_samples}, num_recycles={num_recycles}")
    
    def setup_base_system(
        self, 
        fold_input: folding_input.Input, 
        peptide_chain_id: str
    ) -> None:
        """Setup the base system and compile the model."""
        logger.info("Setting up base system and compiling model...")
        setup_start = time.time()
        
        # Load chemical components dictionary
        ccd = chemical_components.cached_ccd(user_ccd=fold_input.user_ccd)
        
        # Full featurization for the first input
        logger.info("Running full featurization...")
        feat_start = time.time()
        
        featurised_examples = featurisation.featurise_input(
            fold_input=fold_input,
            buckets=None,  # Let it calculate appropriate bucket
            ccd=ccd,
            verbose=True
        )
        
        if not featurised_examples:
            raise ValueError("No featurised examples generated")
        
        self.base_batch = featurised_examples[0]  # Use first example
        feat_time = time.time() - feat_start
        logger.info(f"Featurization completed in {feat_time:.2f} seconds")
        # Stash the original fold_input to enable fresh re-featurisation per variant.
        self.base_fold_input = fold_input
        
        # Initialize model runner with custom config
        logger.info("Initializing model runner...")
        config = make_model_config(
            num_diffusion_samples=self.num_diffusion_samples,
            num_recycles=self.num_recycles,
            return_embeddings=False,
            return_distogram=False
        )
        self.model_runner = AF3ModelRunner(config, self.device, self.model_dir)
        
        # Prefer deriving chain positions from the featurised layout for reliability.
        logger.info(f"Finding positions for peptide chain: {peptide_chain_id}")
        self.peptide_positions = find_chain_token_positions(self.base_batch, peptide_chain_id)
        if self.peptide_positions.size == 0:
            # Fallback to fold_input order/lengths mapping if needed.
            logger.warning("Falling back to fold_input-derived chain mapping for peptide positions.")
            chain_mapping = create_chain_mapping_from_fold_input(fold_input)
            if peptide_chain_id in chain_mapping:
                start_token, end_token = chain_mapping[peptide_chain_id]
                self.peptide_positions = np.arange(start_token, end_token, dtype=np.int32)
        logger.info(f"Found {len(self.peptide_positions)} peptide positions")
        
        
        # Cache peptide MSA row indices where peptide columns are non-gap
        try:
            gap_idx = data_constants.MSA_GAP_IDX
            msa_arr = self.base_batch['msa']  # (msa_rows, num_tokens)
            pep_cols = self.peptide_positions
            if pep_cols is None or len(pep_cols) == 0:
                self.peptide_row_indices = np.array([], dtype=np.int32)
                logger.warning("No peptide positions found; cannot cache peptide MSA rows.")
            else:
                # Prefer rows that contain the full peptide (no gaps) in these columns.
                non_gap_counts = (msa_arr[:, pep_cols] != gap_idx).sum(axis=1)
                pep_len = int(len(pep_cols))
                full_rows = np.where(non_gap_counts == pep_len)[0]

                if full_rows.size > 0:
                    # Expectation for user workflow: peptide MSA is a single query row.
                    chosen = full_rows[:1].astype(np.int32)
                    if full_rows.size > 1:
                        logger.warning(
                            "Multiple MSA rows (%d) contain full non-gap peptide columns; using first row index %d.",
                            full_rows.size, int(chosen[0])
                        )
                else:
                    # Fallback: pick the row with maximum non-gaps among peptide columns, if any.
                    max_count = int(non_gap_counts.max())
                    if max_count == 0:
                        chosen = np.array([], dtype=np.int32)
                        logger.warning(
                            "No non-gap residues detected in peptide columns across MSA rows; cannot cache peptide MSA row."
                        )
                    else:
                        best = int(np.argmax(non_gap_counts))
                        chosen = np.array([best], dtype=np.int32)
                        logger.warning(
                            "No full peptide row found; selected row %d with %d/%d non-gaps for peptide columns.",
                            best, max_count, pep_len
                        )

                self.peptide_row_indices = chosen
                logger.info(f"Cached {len(self.peptide_row_indices)} peptide MSA row(s): {self.peptide_row_indices.tolist()}")
        except Exception as e:
            logger.warning(f"Failed to cache peptide MSA row indices: {e}")
            self.peptide_row_indices = np.array([], dtype=np.int32)
        
        # Compile the model by running inference once
        logger.info("Compiling JAX model (this will take ~80 seconds)...")
        compile_start = time.time()
        
        # Cache the seed used for setup and inference for comparability
        self.setup_seed = fold_input.rng_seeds[0]
        rng_key = jax.random.PRNGKey(self.setup_seed)
        _ = self.model_runner.run_inference(self.base_batch, rng_key)
        
        compile_time = time.time() - compile_start
        self.is_compiled = True
        
        setup_time = time.time() - setup_start
        logger.info(f"Model compilation completed in {compile_time:.2f} seconds")
        logger.info(f"Total setup completed in {setup_time:.2f} seconds")
    
    def run_peptide_variant(
        self,
        peptide_sequence: str,
        peptide_chain_id: str,
    ) -> Dict:
        """Run inference for a peptide variant."""
        if not self.is_compiled:
            raise RuntimeError("Model not compiled. Call setup_base_system first.")
        
        logger.debug(f"Running variant with sequence: {peptide_sequence}")
        variant_start = time.time()
        
        # Update batch for new peptide
        update_start = time.time()
        updated_batch = build_variant_features_with_msa_merge(
            base_batch=self.base_batch,
            base_fold_input=self.base_fold_input,
            peptide_chain_id=peptide_chain_id,
            peptide_sequence=peptide_sequence,
            peptide_positions=self.peptide_positions,
            peptide_row_indices=self.peptide_row_indices,
        )
        update_time = time.time() - update_start
        
        # Run inference with compiled model
        inference_start = time.time()
        # Use the same seed as setup (first seed from input JSON) for comparability
        rng_key = jax.random.PRNGKey(self.setup_seed)
        result = self.model_runner.run_inference(updated_batch, rng_key)
        inference_time = time.time() - inference_start
        
        # Extract results
        extract_start = time.time()
        inference_results = self.model_runner.extract_inference_results(
            batch=updated_batch,
            result=result,
            target_name=f"variant_{peptide_sequence}"
        )
        extract_time = time.time() - extract_start
        
        total_time = time.time() - variant_start
        
        logger.info(
            f"Variant complete in {total_time:.2f}s "
            f"(update: {update_time:.2f}s, inference: {inference_time:.2f}s, "
            f"extract: {extract_time:.2f}s)"
        )
        
        return {
            'sequence': peptide_sequence,
            'inference_results': inference_results,
            'timing': {
                'total': total_time,
                'update': update_time,
                'inference': inference_time,
                'extract': extract_time
            },
            'seed': self.setup_seed,
            'token_chain_ids': inference_results[0].metadata.get('token_chain_ids'),
        }


# Import the actual ModelRunner from run_alphafold.py
import sys
import os
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_THIS_DIR)
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from run_alphafold import ModelRunner as AF3ModelRunner, make_model_config


def _chain_index_mapping(token_chain_ids: Sequence[str]) -> Dict[str, int]:
    """Builds chain_id -> chain_index mapping from token chain ids order of appearance."""
    mapping: Dict[str, int] = {}
    order = []
    for cid in token_chain_ids:
        if cid not in mapping:
            mapping[cid] = len(mapping)
            order.append(cid)
    return mapping


def _compute_best_sample_metrics(
    inference_results: Sequence[model.InferenceResult],
    peptide_chain_id: str,
    receptor_chain_id: str,
) -> Dict[str, float]:
    """Compute metrics for the best-ranking sample among inference_results.

    Returns dict with keys: ranking_score, plddt, pae, iptm.
    - plddt: mean over peptide atoms only.
    - pae: mean over PAE submatrix for peptide tokens only.
    - iptm: chain-pair ipTM between peptide and receptor chains.
    """
    # Pick best by ranking_score
    best = max(inference_results, key=lambda r: float(r.metadata.get('ranking_score', float('nan'))))

    ranking_score = float(best.metadata.get('ranking_score', float('nan')))

    # pLDDT over peptide atoms
    atom_chain_ids = np.asarray(best.predicted_structure.chain_id)
    atom_plddts = np.asarray(best.predicted_structure.atom_b_factor)
    pep_atom_mask = atom_chain_ids == peptide_chain_id
    plddt = float(np.nan) if pep_atom_mask.sum() == 0 else float(np.nanmean(atom_plddts[pep_atom_mask]))

    # PAE peptide-only submatrix
    token_chain_ids = list(best.metadata.get('token_chain_ids', []))
    pep_token_idx = [i for i, cid in enumerate(token_chain_ids) if cid == peptide_chain_id]
    pae = best.numerical_data.get('full_pae')
    if pae is None or len(pep_token_idx) == 0:
        pae_mean = float(np.nan)
    else:
        pae_sub = pae[np.ix_(pep_token_idx, pep_token_idx)]
        pae_mean = float(np.nanmean(pae_sub))

    # ipTM peptide vs receptor
    chain_pair_iptm = best.metadata.get('chain_pair_iptm')
    if chain_pair_iptm is None:
        iptm = float(np.nan)
    else:
        chain_to_index = _chain_index_mapping(token_chain_ids)
        pi = chain_to_index.get(peptide_chain_id)
        ri = chain_to_index.get(receptor_chain_id)
        if pi is None or ri is None:
            iptm = float(np.nan)
        else:
            iptm = float(chain_pair_iptm[pi, ri])

    return {
        'ranking_score': ranking_score,
        'plddt': plddt,
        'pae': pae_mean,
        'iptm': iptm,
    }


def main():
    """Main screening workflow."""
    parser = argparse.ArgumentParser(description="Optimized AlphaFold3 Peptide Screening")
    parser.add_argument('--json_path', type=pathlib.Path, required=True,
                       help='Path to fold input JSON file')
    parser.add_argument('--peptide_chain_id', type=str, required=True,
                       help='Chain ID of the varying peptide chain')
    parser.add_argument('--peptide_sequences', type=pathlib.Path, required=True,
                       help='Path to FASTA file with peptide sequences')
    parser.add_argument('--output_dir', type=pathlib.Path, required=True,
                       help='Output directory for results')
    parser.add_argument('--model_dir', type=pathlib.Path, required=True,
                       help='Path to AlphaFold3 model directory')
    parser.add_argument('--num_diffusion_samples', type=int, default=1,
                       help='Number of diffusion samples to generate (default: 1)')
    parser.add_argument('--num_recycles', type=int, default=3,
                       help='Number of recycles to use during inference (default: 3)')
    parser.add_argument('--receptor_chain_id', type=str, default='A',
                       help='Receptor chain ID for ipTM calculation (default: A)')
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting optimized peptide variant screening")
    logger.info(f"Input: {args.json_path}")
    logger.info(f"Peptide chain: {args.peptide_chain_id}")
    logger.info(f"Sequences: {args.peptide_sequences}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Model config: num_diffusion_samples={args.num_diffusion_samples}, num_recycles={args.num_recycles}")
    
    # Load input data
    fold_input = load_fold_input(args.json_path)
    peptide_sequences = load_peptide_sequences(args.peptide_sequences)
    
    # Initialize optimized runner
    runner = OptimizedPeptideRunner(
        args.model_dir,
        num_diffusion_samples=args.num_diffusion_samples,
        num_recycles=args.num_recycles
    )
    
    # Setup base system (expensive, done once)
    runner.setup_base_system(fold_input, args.peptide_chain_id)
    
    # Process all peptide variants and collect CSV rows
    csv_rows = []
    total_start = time.time()
    
    logger.info(f"Processing {len(peptide_sequences)} peptide variants...")
    
    for i, (name, sequence) in enumerate(peptide_sequences):
        logger.info(f"Processing variant {i+1}/{len(peptide_sequences)}: {name}")
        
        try:
            result = runner.run_peptide_variant(
                sequence,
                args.peptide_chain_id,
            )
            result['name'] = name
            # Write post-processed outputs to variant subdir
            variant_slug = _sanitize_name(name)
            variant_dir = args.output_dir / f"variant_{i+1:04d}_{variant_slug}"
            job_name = f"{variant_slug}"
            variant_summary = write_variant_outputs(
                inference_results=result['inference_results'],
                variant_dir=variant_dir,
                job_name=job_name,
                seed=result['seed'],
            )
            # Compute metrics for best-ranked model
            metrics = _compute_best_sample_metrics(
                result['inference_results'],
                peptide_chain_id=args.peptide_chain_id,
                receptor_chain_id=args.receptor_chain_id,
            )
            # Compose timing as a single string
            t = result['timing']
            timing_str = f"total={t['total']:.2f},update={t['update']:.2f},inference={t['inference']:.2f},extract={t['extract']:.2f}"
            csv_rows.append([
                name,
                result['sequence'],
                result['seed'],
                timing_str,
                variant_dir.as_posix(),
                f"{metrics['ranking_score']:.4f}" if np.isfinite(metrics['ranking_score']) else '',
                f"{metrics['plddt']:.2f}" if np.isfinite(metrics['plddt']) else '',
                f"{metrics['iptm']:.4f}" if np.isfinite(metrics['iptm']) else '',
                f"{metrics['pae']:.2f}" if np.isfinite(metrics['pae']) else '',
            ])
                
        except Exception as e:
            logger.error(f"Failed to process variant {name}: {e}")
            traceback.print_exc()
            raise
            continue
    
    total_time = time.time() - total_start
    # Write CSV summary
    import csv
    summary_csv = args.output_dir / "screening_summary.csv"
    with open(summary_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'name', 'sequence', 'seed', 'timing', 'variant_dir',
            'ranking_score', 'plddt_peptide_mean', 'iptm_peptide_vs_receptor', 'pae_peptide_mean'
        ])
        writer.writerows(csv_rows)
    
    logger.info(f"Screening completed!")
    logger.info(f"Processed {len(csv_rows)}/{len(peptide_sequences)} variants successfully")
    logger.info(f"Total time: {total_time:.2f} seconds")
    avg = total_time/len(csv_rows) if csv_rows else 0.0
    logger.info(f"Average time per variant: {avg:.2f} seconds")
    logger.info(f"Summary CSV saved to: {summary_csv}")


if __name__ == "__main__":
    main()
