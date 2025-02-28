#!/usr/bin/bash 

cp $PDB_DIR/${SDF} $home_data
cp $PDB_DIR/${PDB} $home_data

singularity run --nv /module_dir/gnina.sif gnina -r $home_data/${PDB} -l $home_data/${SDF} --exhaustiveness 64 --minimize --log $home_data/$LOGFILE   
# mv $home_data/$LOGFILE $PDB_DIR

# Check if $LOGFILE was generated
if [[ -e "${home_data}/$LOGFILE" ]]; then
    # Extract RMSD and Affinity values from $LOGFILE
    rmsd=$(grep "RMSD:" ${home_data}/$LOGFILE | awk '{print $2}')
    affinity=$(grep "Affinity:" ${home_data}/$LOGFILE | awk '{print $2}')
    var=$(grep "Affinity:" ${home_data}/$LOGFILE | awk '{print $3}')
    cnnscore=$(grep "CNNscore:" ${home_data}/$LOGFILE | awk '{print $2}')
    cnnaff=$(grep "CNNaffinity:" ${home_data}/$LOGFILE | awk '{print $2}')
    cnnvar=$(grep "CNNvariance:" ${home_data}/$LOGFILE | awk '{print $2}')

    # Append the values to the CSV file
    echo "$rmsd,$affinity,$var,$cnnscore,$cnnaff,$cnnvar"
fi

