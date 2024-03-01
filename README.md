# Automatic and Personalized Tunnel Book Generation from Photographs
A tunnel book is a three-dimensional, layered piece of art that is created by making a "tunnel" ("accordion") structure out of a series of cut or stacked layers. It incorporates a perspective view, creating the illusion of depth and space when the book is opened and the layers are expanded.


## Description
The goal of this project is to automate the process of creating a tunnel book by suggesting optimal cutting positions based on photograph content. We aim to define a "tunnel-bookiness" metric, assessing the degree to which an image lends itself to this artistic form, considering factors like perspective, layering, and visual appeal. The project aims to integrate image processing and machine learning techniques (e.g., depth estimation [2], inpainting[1]) to assist users/artists in creating tunnel books from still 2D photographs.


## Tasks
- Perform a literature review
- Implement a baseline method to suggest the cut/layers from a single image/photograph
- Quantify the practical feasibility of suggested layers/cut
    - consider physical constraints such as object positioning, absence of floating objects, how much the suggested cut can hold vertically, etc
- Assess the deviation/distortion between the suggested layered depth from the cuts and the estimated depth
- Define and quantify other physical constraints of a tunnel book numerically from the suggested cut
- Introduce a "tunnel-bookiness" metric indicating whether an image is suitable for the tunnel book form of art
- Formalize the problem of automatic cut suggestion as an optimization problem, aiming to minimize constraints
    - e.g., the practicality of the cut, and distortion between the original depth and the one resulting from the cut
- Improve the baseline method


## Deliverables
Code, well cleaned up and easily reproducible. Written Report, explaining the literature and steps taken for the project


## References
**Remark:** these papers are ment to only serve as a starting point for possible machine learning techniques. It is up to us to find image processing techniques and dig deeper into the machine learning techniques.

- [RePaint: Inpainting using Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2201.09865)
    - [Github repo](https://github.com/andreas128/RePaint)
- [Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer](https://arxiv.org/abs/1907.01341)
    - [Github repo](https://github.com/isl-org/MiDaS)

