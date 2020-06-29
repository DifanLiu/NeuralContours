# Data Format

This page includes the information about precomputed lines and geometric features.

### Suggestive Contours
- `suggestive_contour.png`: unfiltered suggestive contours. In `rtsc-1.6/rtsc.cc`, set `sug_thresh = 0` and `draw_sc = 1`.
- `nv.png`: normal dot view direction. In `rtsc-1.6/rtsc.cc`, check variable `ndotv` in function `compute_perview`.
- `suggestive_contour_feature.png`: `dwkr` (derivative of the radial curvature) which is normalized by the 90% percentile maximum and saved as an image. In `rtsc-1.6/rtsc.cc`, check variable `sctest_num` in function `compute_perview`.
- `suggestive_contour_info.txt`: the first number is the 90% percentile maximum of `dwkr` (negative `dwkr` are first removed). The second number is feature size of the mesh, in `rtsc-1.6/rtsc.cc`, check variable `feature_size`.

### Ridges
- `ridge.png`: unfiltered ridges. In `rtsc-1.6/rtsc.cc`, set `rv_thresh = 0` and `draw_ridges = 1`.
- `ridge_feature.png`: `k1` (the positive first principal curvature) which is normalized by the 90% percentile maximum and saved as an image. In `rtsc-1.6/rtsc.cc`, check variable `themesh->curv1`.
- `ridge_info.txt`: the first number is the 90% percentile maximum of `themesh->curv1` (negative `themesh->curv1` are first removed). The second number is feature size of the mesh.

### Valleys
- `valley.png`: unfiltered valleys. In `rtsc-1.6/rtsc.cc`, set `rv_thresh = 0` and `draw_valleys = 1`.
- `valley_feature.png`: `k2` (the negative first principal curvature) which is normalized by the 90% percentile maximum and saved as an image. In `rtsc-1.6/rtsc.cc`, check variable `themesh->curv1`.
- `valley_info.txt`: the first number is the 90% percentile maximum of `-themesh->curv1` (positive `themesh->curv1` are first removed). The second number is feature size of the mesh.

### Apparent Ridges
- `apparent_ridge.png`: unfiltered apparent ridges. In `rtsc-1.6/rtsc.cc`, set `ar_thresh = 0` and `draw_apparent = 1`.
- `apparent_ridge_feature.png`: `kt` (view-dependent curvature) which is normalized by the 90% percentile maximum and saved as an image. In `rtsc-1.6/rtsc.cc`, check variable `q1` in function `compute_perview`.
- `apparent_ridge_info.txt`: the first number is the 90% percentile maximum of `kt` (negative `kt` are first removed). The second number is feature size of the mesh.

### Contours and Creases
- `base.png`: contours and creases rendered by Blender Freestyle under default setting.

### View-based Shape Representations
- `smooth_*.png`: shaded rendering. We set `currsmooth = smooth_value * themesh->feature_size()` in `rtsc-1.6/rtsc.cc`. Normals are diffused by function `filter_normals` in `rtsc-1.6/rtsc.cc`. From `smooth_1.png` to `smooth_5.png`, we set `smooth_value = [1， 2， 3， 4， 5]`.
- `depth.png`: depth image.