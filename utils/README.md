# Utils

*  `conda/` holds files to create the respective conda environments
*  `download_MRI_data.py` downloads all the MRI data of the Duke-Breast-Cancer-MRI cohort from the TCIA
*  `download_tabular_data.sh` downloads the annotation boxes and clinical data from the TCIA
*  `download_pretrained_weihts.sh` downloads ImageNet pretrained weights of the MAE model
*  `generate_patches/` contains files to save MR patches as hdf5 file
*  `split_data.py` splits data into train, validation and test

`bboxes/` contains files to generate bounding boxes of the chest region.
If you want to run the scripts you have to generate breast tissue segmentation maps with help of the
[model provided with the data](https://github.com/mazurowski-lab/3D-Breast-FGT-and-Blood-Vessel-Segmentation).
However, bounding boxes generated in this way are also provided at `/data/bboxes/`.
