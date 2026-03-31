#!/usr/bin/env bash

json_path="fold_input.json"
output_dir="."

#model_dir
model_dir="/folding/data/params"

#db_dir
db_dir="/folding/data"

#af3_dir
af3_dir="/home/hehuang/Tools/alphafold3"

# Small BFD database path, used for protein MSA search.
# previous download with unknown version
small_bfd_database_path="$db_dir/small_bfd/bfd-first_non_consensus_sequences.fasta"

# Mgnify database path, used for protein MSA search.
# previous download with up-to-date version
mgnify_database_path="$db_dir/mgnify/mgy_clusters_2022_05.fa"

#UniProt database path, used for protein paired MSA search.
# previous download with old version. up-to-date version is uniprot_all_2021_04.fa
uniprot_cluster_annot_database_path="$db_dir/uniprot/uniprot.fasta"

# UniRef90 database path, used for MSA search. The MSA obtained by searching it is used to construct the profile for template search.
# previous download with unknown version. up-to-date version is uniref90_2022_05.fa
uniref90_database_path="$db_dir/uniref90/uniref90.fasta"

#NT-RNA database path, used for RNA MSA search.
# missing
ntrna_database_path="$db_dir/nt_rna_2023_02_23_clust_seq_id_90_cov_80_rep_seq.fasta"

#Rfam database path, used for RNA MSA search.
# missing
rfam_database_path="$db_dir/rfam_14_9_clust_seq_id_90_cov_80_rep_seq.fasta"

#RNAcentral database path, used for RNA MSA search.
# missing
rna_central_database_path="$db_dir/rnacentral_active_seq_id_90_cov_80_linclust.fasta"

#PDB database directory with mmCIF files path, used for template search.
# previous download with unknown version
pdb_database_path="$db_dir/pdb_mmcif/mmcif_files"

#PDB sequence database path, used for template search.
# previous download with unknown version. up-to-date version is pdb_seqres_2022_09_28.fasta"
seqres_database_path="$db_dir/pdb_seqres/2.3pdb_seqres.txt"

python $af3_dir/run_alphafold.py \
    --json_path=$json_path \
    --output_dir=$output_dir \
    --model_dir=$model_dir \
    --db_dir=$db_dir \
    --small_bfd_database_path=$small_bfd_database_path \
    --mgnify_database_path=$mgnify_database_path \
    --uniprot_cluster_annot_database_path=$uniprot_cluster_annot_database_path \
    --uniref90_database_path=$uniref90_database_path \
    --ntrna_database_path=$ntrna_database_path \
    --rfam_database_path=$rfam_database_path \
    --rna_central_database_path=$rna_central_database_path \
    --pdb_database_path=$pdb_database_path \
    --seqres_database_path=$seqres_database_path
