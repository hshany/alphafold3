#!/usr/bin/env python3
"""
Protein Sequence Alignment Script

This script:
1. Loads a query protein sequence from a string
2. Loads a template protein sequence from a CIF file
3. Aligns the two sequences
4. Outputs the mapping of residues as 0-based indices
"""

import json
import numpy as np
from Bio import pairwise2
from Bio.PDB import MMCIFParser
from Bio.PDB.Polypeptide import is_aa
from Bio.SeqUtils import seq1

def extract_sequence_from_cif(cif_file, chain_id='A'):
    """
    Extract the amino acid sequence from a CIF file for a specific chain.
    
    Args:
        cif_file (str): Path to the CIF file
        chain_id (str): Chain identifier (default: 'A')
        
    Returns:
        str: Amino acid sequence in one-letter code
    """
    # Parse the CIF file
    parser = MMCIFParser()
    structure = parser.get_structure('template', cif_file)
    
    # Extract the sequence for the specified chain
    sequence = ""
    residue_ids = []
    
    # Get the first model in the structure
    model = structure[0]
    
    # Check if the chain exists
    if chain_id not in model:
        raise ValueError(f"Chain {chain_id} not found in the CIF file")
    
    # Iterate through residues in the chain
    for residue in model[chain_id]:
        # Check if it's an amino acid and not a hetero-residue
        if is_aa(residue.get_resname(), standard=True) and not residue.id[0].strip():
            try:
                # Convert three-letter code to one-letter code
                one_letter = seq1(residue.get_resname())
                sequence += one_letter
                residue_ids.append(residue.id[1])  # Store residue number
            except Exception as e:
                print(f"Warning: Could not convert residue {residue.get_resname()}: {e}")
    
    return sequence, residue_ids

def align_sequences(query_seq, template_seq):
    """
    Align two protein sequences using global alignment.
    
    Args:
        query_seq (str): Query protein sequence
        template_seq (str): Template protein sequence
        
    Returns:
        tuple: Aligned sequences and alignment score
    """
    # Perform global alignment
    alignments = pairwise2.align.globalms(
        query_seq, 
        template_seq,
        2,      # Match score
        -1,     # Mismatch penalty
        -2,     # Gap opening penalty
        -0.5    # Gap extension penalty
    )
    
    # Get the best alignment
    best_alignment = alignments[0]
    aligned_query, aligned_template, score = best_alignment.seqA, best_alignment.seqB, best_alignment.score
    
    return aligned_query, aligned_template, score

def get_residue_mapping(aligned_query, aligned_template):
    """
    Get mapping between residues in the aligned sequences.
    
    Args:
        aligned_query (str): Aligned query sequence with gaps
        aligned_template (str): Aligned template sequence with gaps
        
    Returns:
        tuple: Two lists containing the 0-based indices of mapped residues
    """
    query_indices = []
    template_indices = []
    
    query_pos = 0
    template_pos = 0
    
    # Iterate through the aligned sequences
    for i in range(len(aligned_query)):
        # Skip if both positions are gaps
        if aligned_query[i] == '-' and aligned_template[i] == '-':
            continue
            
        # If there's a match or mismatch (not a gap in either sequence)
        if aligned_query[i] != '-' and aligned_template[i] != '-':
            query_indices.append(query_pos)
            template_indices.append(template_pos)
            
        # Update positions
        if aligned_query[i] != '-':
            query_pos += 1
            
        if aligned_template[i] != '-':
            template_pos += 1
    
    return query_indices, template_indices

def main():
    # 1. Load query protein sequence from a string
    query_sequence = "SMPSWQLALWATAYLALVLVAVTGNAIVIWIILAHRRMRTVTNYFIVNLALADLCMAAFNAAFNFVYASHNIWYFGRAFCYFQNLFPITAMFVSIYSMTAIAADRYMAIVHPFQPRLSAPSTKAVIAGIWLVALALASPQCFYSTVTMDQGATKCVVAWPEDSGGKTLLLYHLVVIALIYFLPLAVMFVAYSVIGLTLWRRAVPGHQAHGANLRHLQAMKKFVKTMVLVVLTFAICWLPYHLYFILGSFQEDIYCHKFIQQVYLALFWLAMSSTMYNPIIYCCLNHRF"
    print(f"Query sequence length: {len(query_sequence)}")
    
    # 2. Load template protein sequence from CIF file
    cif_file = "../../7xwo.cif"  # Change this to your CIF file path
    chain_id = "B"

    try:
        template_sequence, residue_ids = extract_sequence_from_cif(cif_file, chain_id=chain_id)
        print(f"Template sequence length: {len(template_sequence)}")
    except Exception as e:
        print(f"Error loading CIF file: {e}")
        return
    
    # 3. Align the sequences
    aligned_query, aligned_template, score = align_sequences(query_sequence, template_sequence)
    print(f"Alignment score: {score}")
    print(f"Aligned query:    {aligned_query[:]}...")
    print(f"Aligned template: {aligned_template[:]}...")
    
    # 4. Get the residue mapping and output as 0-based indices
    query_indices, template_indices = get_residue_mapping(aligned_query, aligned_template)
    
    # Create the output in the requested format
    mapping = {
        "queryIndices": query_indices,
        "templateIndices": template_indices
    }

    # Output as JSON (as one-liners)
    print("\nResidue Mapping:")
    print(f'"queryIndices": {query_indices},')
    print(f'"templateIndices": {template_indices}')

    # Optional: Save to file
    with open("residue_mapping.json", "w") as f:
        json.dump(mapping, f, separators=(',', ':'))
    
if __name__ == "__main__":
    main()
