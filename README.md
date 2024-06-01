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
layer#.npy: an array of object masks on one layer (with the same size as the downsampled input image).

# Tunnel Book Generation Parameters
- filter_segma: Gaussian filter parameter which can smooth the mask edge. Default: 10
- filter_shreshold: Filter parameter, which decide the size and shape of the mask. Default: 0.2
- link_to_ground: Whether to link the layers to the ground. Default: False
- sample_method: the sample methods for re-segmentation. Grid method('grid') and method based on superpixel('superpixel') are implemented. Default: 'superpixel'
- Filter_layer: Whether to use the filted mask as output or the original mask.

# Start

Please find the tunnel_book.ipynb to get start with our project.

Please find the notebook inside inpaint folder to run detailed example or automatic example.

# Parallax Effect Web
Click [here](https://koitu.github.io/cs413-computational-photography-project)


# To update the Github pages
```
git push origin --delete gh-pages
git subtree push --prefix website origin gh-pages
```
