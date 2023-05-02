#!/bin/bash

# run with the duke_sgmt conda environment

root_dir=$( dirname `dirname $PWD` )

script="$root_dir/utils/segmentation/3D-Breast-FGT-and-Blood-Vessel-Segmentation/predict.py"
data_dir="$root_dir/data/TCIA/Duke-Breast-Cancer-MRI"

modality="t1_FS"
out_folder="SEG"

rm -rf .tmp/

for dir in $data_dir/Breast*/
do
    echo $( basename $dir )
    python ./segment.py $dir$modality $script $dir$out_folder
    rm -rf .tmp/
done
