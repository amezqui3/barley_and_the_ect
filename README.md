# The shape of things to come

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/amezqui3/ect_and_barley/HEAD)

## Quantifying the shape of plants with the Euler characteristic

The jupyter notebook `jupyter/tutorial_shape_of_things_to_come` is meant to be a first step introduction to compute [Euler Characteristic Curves](https://doi.org/10.1016/j.patrec.2014.07.001) (ECC) and the [Euler Characteristic Transform](https://doi.org/10.1093/imaiai/iau011) (ECT) for 2D and 3D images.  This repository includes the jupyter notebooks for the [2021 Annual Meeting of the NAPPN](https://www.nappn2021.org/agenda) to be held virtually, led by [Erik Amézquita](http://egr.msu.edu/~amezqui3). This introduction is tailored for a phenotyping plant biology audience, so the focus is on topological tools for image analysis.

These slides are written to be presented as [RISE slides](https://rise.readthedocs.io/en/stable/index.html), however the notebook should be self contained without needing this installed. If you see a lot of weird cell toolbars in the notebook (which are used for controlling the slideshow version), these can be removed from your view of the jupyter notebook by going to View &rarr; Cell Toolbar &rarr; None

The relevant python functions to compute the ECCs and ECT are found in `jupyter/brewing_utils.py`. I'm still working to make this an actual package. The utils file contain other image processing-related functions relevant to clean X-ray CT scans of barley spikes.

## Contents

- `data`
    - `directions`: plots of directions sampled on the sphere
    - `tiffs`: sample 2D and 3D TIFF image files
- `ects`
    - `*.csv`: Computed ECT values for more than 3,000 barley seeds from 28 different barley lines (land races)
    - `*.rds`: R-native file with the classification results of a trained SVM
- `jupyter`:
    - `complexify_binary_image.ipynb`: (python) Details on how to get a 2D dual cubical complex from a grayscale image
    - `ect_computation.ipynb`: (python) Details on how to compute ECCs and the ECT of a grayscale image
    - `shape_descriptor_analysis.ipynb`: (R) Compute and plot classification accuracy of traditional vs topological shape descriptors
    - `shape_descriptor_classification.ipynb`: (R) Train an SVM to classify 28 different barley lines based solely on the shape of their grains.
    - `sphere_directions.ipynb`: (python) Details on how to define 3D directions either uniformly or randomly placed.
    - `tutorial_shape_of_things_to_come.ipynb` (python) A gentle introduction to TDA and the ECT. Written originally for the [2021 Annual Meeting of the NAPPN](https://www.nappn2021.org).
    - `brewing_utils.py` Python file defining the relevant functions to compute the ECT (plus a number of extra image processing stuff that I need to clean up. Refer to the relevant detailed notebook first.)
    - `descriptor_analysis_utils.R`. R functions to analyze and plot SVM results. Extremely ad-hoc. Refer to the relevant detailed notebook first.
- `plots`: Plots generated by the relevant R data analysis

## To run locally

If you want to run locally all the notebooks, make sure you have `jupyter` enabled for **both** python and R.

You will also need to have the following python libraries

     matplotlib scipy numpy tifffile pandas

And the following R libraries

     ggplot2 ggdendro e1071 reshape2 dplyr viridis

