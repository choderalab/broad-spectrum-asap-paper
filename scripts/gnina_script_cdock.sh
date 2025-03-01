#!/usr/bin/bash 

cp $PDB_DIR/${SDF} $home_data
cp $PDB_DIR/${PDB} $home_data

# Seed for crossdocking CNN model
CNN_INDICES=(1 2 3 4)

for i in "${CNN_INDICES[@]}"; do
    echo $PDB_DIR/${label}_a_ligand.sdf for crossdock model seed $i
    # Initialize sums and count
    sum_rmsd=0
    sum_affinity=0
    sum_var=0
    sum_cnnscore=0
    sum_cnnaff=0
    sum_cnnaff_sq=0  # Sum of squares for variance calculation
    count=0
    singularity run --nv /module_dir/gnina.sif gnina -r $home_data/${PDB} -l $home_data/${SDF} --exhaustiveness 64 --minimize --log $home_data/$LOGFILE --scoring vina --cnn "crossdock_default2018_${i}" 

    # Check if $LOGFILE was generated
    if [[ -e "${home_data}/$LOGFILE" ]]; then
        # Extract RMSD and Affinity values from $LOGFILE
        rmsd=$(grep "RMSD:" ${home_data}/$LOGFILE | awk '{print $2}')
        affinity=$(grep "Affinity:" ${home_data}/$LOGFILE | awk '{print $2}')
        var=$(grep "Affinity:" ${home_data}/$LOGFILE | awk '{print $3}')
        cnnscore=$(grep "CNNscore:" ${home_data}/$LOGFILE | awk '{print $2}')
        cnnaff=$(grep "CNNaffinity:" ${home_data}/$LOGFILE | awk '{print $2}')

        sum_rmsd=$(echo "$sum_rmsd + $rmsd" | bc)
        sum_affinity=$(echo "$sum_affinity + $affinity" | bc)
        sum_var=$(echo "$sum_var + $var" | bc)
        sum_cnnscore=$(echo "$sum_cnnscore + $cnnscore" | bc)
        sum_cnnaff=$(echo "$sum_cnnaff + $cnnaff" | bc)
	sum_cnnaff_sq=$(echo "$sum_cnnaff_sq + ($cnnaff * $cnnaff)" | bc)  # Sum of squares
        ((count++))
    else
        echo "$LOGFILE not found for $pdb_file"
    fi
done
# Compute averages if count > 0
if ((count > 0)); then
    avg_rmsd=$(echo "scale=4; $sum_rmsd / $count" | bc)
    avg_affinity=$(echo "scale=4; $sum_affinity / $count" | bc)
    avg_var=$(echo "scale=4; $sum_var / $count" | bc)
    avg_cnnscore=$(echo "scale=4; $sum_cnnscore / $count" | bc)
    avg_cnnaff=$(echo "scale=4; $sum_cnnaff / $count" | bc)
    # Compute variance
    var_cnnaff=$(echo "scale=4; ($sum_cnnaff_sq / $count) - ($avg_cnnaff * $avg_cnnaff)" | bc)

    # Append the values to the CSV file
    echo "$avg_rmsd,$avg_affinity,$avg_var,$avg_cnnscore,$avg_cnnaff,$var_cnnaff" 
fi


