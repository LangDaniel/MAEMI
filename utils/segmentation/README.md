# Segmentation

Clone the segmentation model from the
[github repo](https://github.com/mazurowski-lab/3D-Breast-FGT-and-Blood-Vessel-Segmentation)
and run `main.sh` in the `duke_sgmt` conda environment.

The following fixes have been applied for `3D-Breast-FGT-and-Blood-Vessel-Segmentation` to work:
In `predict.py` specify the arguments of `pred_and_save_masks_3d_simple()`:
```
pred_and_save_masks_3d_simple(
    saved_model_path=args.model_save_path,
    dataset=dataset,
    unet=unet,
    n_classes=n_classes,
    n_channels=1,
    save_masks_dir=args.save_masks_dir,
)   
```

In `model_utils.py` remove `mask = batch['mask']` in 
line 747 and `mask = mask.to(device, dtype=torch.float32)` in line 750.
