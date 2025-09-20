#!/usr/bin/env python3

import json
import sys
from collections import defaultdict

def parse_af3_scores(json_file):
    """Parse AF3 confidence JSON and report average pLDDT for each chain."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    chain_ids = data['atom_chain_ids']
    plddts = data['atom_plddts']
    
    # Group pLDDT scores by chain
    chain_scores = defaultdict(list)
    for chain_id, plddt in zip(chain_ids, plddts):
        chain_scores[chain_id].append(plddt)
    
    # Calculate and print average pLDDT for each chain
    print("Chain\tAverage pLDDT")
    print("-" * 20)
    for chain_id in sorted(chain_scores.keys()):
        avg_plddt = sum(chain_scores[chain_id]) / len(chain_scores[chain_id])
        print(f"{chain_id}\t{avg_plddt:.2f}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python parse_af3_scores.py <confidence_file.json>")
        sys.exit(1)
    
    parse_af3_scores(sys.argv[1])