# Training

Run `train.sh <PATH_TO_PARAMETERFILE>` to train the model, with
the parameters of the paper given in `./parameter/par.yml`.

*  `mae3d/` contains the developed 3D masked autoencoder model
*  `sequence.py` contains a Dataset to read MRI patches

For the `mae` conda environment provided, the `timm`
[fix](https://github.com/huggingface/pytorch-image-models/issues/420#issuecomment-776459842)
specified in the original
[MAE repo](https://github.com/facebookresearch/mae) has to be applied.
