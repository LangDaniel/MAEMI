# Patches

scripts have to be run in the following order:

*  `retrieve_file_paths.py` stores the path of the downloades files as a CSV file
*  `image_patches.py` generates the MR patch hdf5 file
*  `ROI_sgmt.py` adds bbox ROI segmentations to the file.
