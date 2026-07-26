# benthoz
The BENTHOZ-2015 data set is a collection of around 10,000 images of the sea floor in coastal Australia, taken by an Autonomous Underwater Vehicle (AUV). It includes sensor data from the AUV, as well as expert annotations on over 400,000 image points.
The image points are labbelled according to the CATAMI hierarchy, conforming to a taxonomy of nearly 150 classes.
For more details on the data set and broader context, please read the data set descriptor on the Nature Publishing Group's Scientific Data journal (http://www.nature.com/articles/sdata201557). If you make use of the data set or this toolbox in an academic context, that is the paper to cite.

This repository consists of a growing set of tools to access, manipulate, and perform automated classification on data in the BENTHOZ-2015 data set. The primary aim is to reduce the barrier to entry in conducting research, enabling a wider range of machine learning and computer vision researchers to make progress on the automated classification of benthic imagery.

## Downloading the Data Set
The official copy of the point metadata and labels is available on Figshare http://dx.doi.org/10.6084/m9.figshare.1524139.
The simplest way to acquire the images is to follow the instructions on Squidle at https://squidle.acfr.usyd.edu.au/benthoz2015. It involves downloading a list of images, and the image downloader tool for your platform.

## Data Set Splits
The data_splits folder contains labels for the training, calibration and test sets used in my PhD thesis (http://hdl.handle.net/2123/14511).

## Getting Started
Needs Python 3.14 or newer.
```bash
pip install -e ".[dev,notebook]"
ln -s /path/to/benthoz_data benthoz_data   # the imagery is not in the repo
pytest
```
The hierarchy plot in the notebook additionally needs graphviz, which has to be installed
at the system level before the Python package will build:
```bash
sudo apt install graphviz graphviz-dev && pip install -e ".[graphviz]"
```
Then work through `prepare_dataset.ipynb`, which downloads the imagery, builds the CATAMI class
hierarchy, and writes the point labels for each split to parquet.

Alternatively `docker build -t benthoz .` gives you the whole stack including GPU support.

## Labels
Annotations are sparse points, roughly 40 per image, stored as one parquet file per split
(~1.4 MB for all three). Dense `(n_classes, height, width)` masks are built on demand with
`prep.labels.points_to_mask` rather than stored on disk. See CLAUDE.md for the hierarchy
semantics and the gotchas.
