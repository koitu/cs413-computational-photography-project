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

# Parallax Effect Web
Click [here](https://koitu.github.io/cs413-computational-photography-project)

# To update the Github pages
```
git push origin --delete gh-pages
git subtree push --prefix website origin gh-pages
```
