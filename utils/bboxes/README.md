# Bounding Boxes

Use `bboxes_ROI.py` to adjust the tumor bounding boxes provided with the data.
Run `bboxes_chest_from_sgmt.py` to generate the bounding boxes from segmentations
generated with the model from
[the official repository](https://github.com/mazurowski-lab/3D-Breast-FGT-and-Blood-Vessel-Segmentation)
and `adjust_bboxes.py` to adjust them such that no tumor regions lay
outside the bboxes.
