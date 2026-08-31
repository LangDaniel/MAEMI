# Inference

Turns a trained 3D MAE checkpoint into per-case anomaly maps, and builds
the DCE-derived reference maps they're evaluated against.

## `mae_predict.py`

```bash
python mae_predict.py <ckpt> <data_dir> <label_file>
```

* `ckpt` — path to a model checkpoint inside a training output folder.
  That folder must also contain `parameter.yml` and `sequence.py`,
  written there automatically by [`training/train.py`](./../training/train.py)
  at the start of training, so the exact data/model config and dataset
  class used for training can be reloaded here.
* `data_dir` — root folder containing `patches.h5` (see
  [`utils/generate_patches/`](./../utils/generate_patches)).
* `label_file` — CSV of case identifiers to run inference on.

For each case, the full (left + right breast) volume is tiled into
overlapping, model-sized patches via the sliding-window
[`SliceIterator`](./slices.py) (`stride` and `repeat` are set at the top
of the script). Every patch is passed through the model with a fixed
`mask_ratio` (loaded from `parameter.yml`), and the squared error between
each masked patch and its reconstruction becomes that patch's anomaly
score. Overlapping patches are averaged, the whole tiling is repeated
`repeat` times with fresh random masks and averaged again, and the result
is smoothed with a `(3, 3, 2)` minimum filter.

## `slices.py`

`SliceIterator(vol_shape, patch_size, stride, offset)`: a generator over
sliding-window slice tuples covering a volume, used by `mae_predict.py`
to tile a full breast volume into model-input-sized patches with a given
stride.

## `subtraction_images.py`

Builds the evaluation reference: for each case it computes a DCE
subtraction map (mean squared difference between the non-contrast
reference sequence and the post-contrast phases) plus the tumor bounding
box and breast-tissue segmentation in the same coordinate frame, and
writes them to `prediction/contrast_maps/<label_file_stem>.h5` alongside
the anomaly maps from `mae_predict.py` for comparison.