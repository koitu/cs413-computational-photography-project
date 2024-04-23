# Automatic and Personalized Tunnel Book Generation from Photographs
clone this repo recursively
```
git clone REPO_URL --recursive
```
otherwise you will need to add submodules manually
```
git submodule update --init --recursive
```

# Segmented object Output
layers_idx: list of list. 
```[[1, 7], [2, 4, 11, 12], [0, 3, 5, 6, 8, 9, 10]]```

layers: list of segment-anything mask (downsampling of the input image)<br>
```
layer[0] = 
[{'segmentation': array([[False, False, False, ..., False, False, False],
         [False, False, False, ..., False, False, False],
         [False, False, False, ..., False, False, False],
         ...,
         [False,  True,  True, ...,  True,  True, False],
         [False, False,  True, ...,  True,  True, False],
         [False, False,  True, ...,  True,  True, False]]),
  'area': 58961,
  'bbox': [0, 574, 516, 209],
  'predicted_iou': 1.0113445520401,
  'point_coords': [[40.46875, 673.75]],
  'stability_score': 0.9863142967224121,
  'crop_box': [0, 0, 518, 784]},
 {'segmentation': array([[False, False, False, ..., False, False, False],
         [False, False, False, ..., False, False, False],
         [False, False, False, ..., False, False, False],
         ...,
         [False, False, False, ..., False, False, False],
         [False, False, False, ..., False, False, False],
         [False, False, False, ..., False, False, False]]),
  'area': 4893,
  'bbox': [438, 638, 79, 82],
  'predicted_iou': 0.9729478359222412,
  'point_coords': [[477.53125, 673.75]],
  'stability_score': 0.9730056524276733,
  'crop_box': [0, 0, 518, 784]}]
```


