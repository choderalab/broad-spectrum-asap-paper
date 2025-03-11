#!/bin/bash

ref_pdb="input_files/8dz0.pdb"
csv_fn="data/sars2_alignment/SARS-CoV-2-Mpro.csv"
template_dir="data/template_dir_apo"
out_dir="data/sars2_alignment/cf_results_human_CoV/"

mkdir $template_dir
cp $ref_pdb ${template_dir}/0001.pdb 

module load colabfold/v1.5.2

echo "Running ColabFold"
colabfold_batch $csv_fn $out_dir  \
 --num-recycle 3 --num-models 3 --model-type "alphafold2_multimer_v3" \
 --templates --custom-template-path $template_dir \
