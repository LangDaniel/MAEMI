# Inference

*  `mae_predict.py <CKPT> <DATADIR> <LABELFILE>`  
   Generates the anomaly maps for cases stored in <LABELFILE> which have
   to be present in <DATADIR>, with the trained model checkpoint specified by
   <CKPT>.  

*  `slices.py`
   Module to provide indices used to iterate over whole MRI-images via patches

*  `subtraction_images.py`
   Generates the subtraction images.
   TODO: specify path with argparser, currently test files are hard coded
