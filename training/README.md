# Training

## `train.sh`

```bash
./train.sh <PATH_TO_PARAMETER_FILE_OR_DIR> [-c CPU_LIST] [-fg]
```

`pdir` may be a single `.yml` file or a directory of them: `train.py` will run each
one in turn, logging and skipping any run that raises rather than aborting
the whole batch. `-c/--cpu-list` sets the CPU range for `taskset`
(default `0-10`); without `-fg` the run is detached with `nohup` and its
PID appended to `./log`.

The parameters used for the paper are in [`parameter/par.yml`](./parameter/par.yml).

Each run creates a fresh, auto-numbered output folder
(`<output.output_dir>/<output.sub>_<n>`) and freezes a copy of
`train.py`, `sequence.py`, and the parameter file (as `parameter.yml`).
Later, 
[`inference/mae_predict.py`](./../inference/mae_predict.py) reloads those files
to reconstruct the exact model/data config a checkpoint was trained with.

## `sequence.py`

`(torch.utils.data.Dataset)` that reads breast MRI patches out of
the HDF5 file produced by
[`utils/generate_patches/`](./../utils/generate_patches): for each case
it loads every configured modality/channel, crops (randomly during
training, centered otherwise) to the configured patch shape, applies the
configured augmentations (flip, rotate, zoom, offset, blur, noise), and
returns a `(C, H, W, D)` tensor plus a label.

## `mae3d/`

The 3D MAE model, training engine, and entry points
(`main_pretrain.py`, `main_finetune.py`, `main_linprobe.py`) — see
[`mae3d/README.md`](./mae3d/README.md).

For the `timm` version used, the
[fix](https://github.com/huggingface/pytorch-image-models/issues/420#issuecomment-776459842)
specified in the original
[MAE repo](https://github.com/facebookresearch/mae) has to be applied.
