# Data Format

This page includes the information about precomputed lines and geometric features.

### Suggestive Contours
- `suggestive_contour.png`: unfiltered suggestive contours. In `rtsc-1.6/rtsc.cc`, set `sug_thresh = 0` and `draw_sc = 1`.
- `suggestive_contour_feature.png`: `dwkr` (derivative of the radial curvature) which is normalized by the 90% percentile maximum and saved as an image. In `rtsc-1.6/rtsc.cc`, check variable `sctest_num` in function `compute_perview`.
- `suggestive_contour_info.txt`: the first number is the 90% percentile maximum of `dwkr` (vertices with negative `dwkr` are first removed). The second number is feature size of the mesh, in `rtsc-1.6/rtsc.cc`, check variable `feature_size`.

### Ridges
- `ridge.png`: unfiltered ridges. In `rtsc-1.6/rtsc.cc`, set `rv_thresh = 0` and `draw_ridges = 1`.
- `ridge_feature.png`: `k1` (the positive first principal curvature) which is normalized by the 90% percentile maximum and saved as an image. In `rtsc-1.6/rtsc.cc`, check variable `themesh->curv1`.
- `ridge_info.txt`: the first number is the 90% percentile maximum of `themesh->curv1` (vertices with negative `themesh->curv1` are first removed). The second number is feature size of the mesh.