# [Unsupervised Domain Adaptation in the dissimilarity space for Person Re-identification](https://arxiv.org/abs/2007.13890 "Unsupervised Domain Adaptation in the dissimilarity space for Person Re-identification")

## Installation


Make sure `conda <https://www.anaconda.com/distribution/>`_ is installed.

```
    git clone https://github.com/djidje/D-MMD

    # create environment
    cd D-MMD
    conda create --name d-mmd python=3.7
    conda activate d-mmd

    # install dependencies
    pip install -r requirements.txt

    # install torch and torchvision (select the proper cuda version to suit your machine)
    conda install pytorch torchvision cudatoolkit=9.0 -c pytorch

    # install torchreid (don't need to re-build it if you modify the source code)
    python setup.py develop
```

## To reproduce experiments :

### 0. Preparation of data
The code is inspired from: 
https://github.com/KaiyangZhou/deep-person-reid

**Please arrange the data as proposed here:
**
https://kaiyangzhou.github.io/deep-person-reid/datasets.html

### 1. Train source domain

To train a model based on source:
```
    python source_training.py
```

You can run it for Market1501, DukeMTMC and MSMT17 by changing the source in the python file by their correspunding names : *market1501, dukemtmcreid and msmt17* :

```python
	source = 'market1501'
	target = source
```
The model will be saved in this repo and will be used to perform the adaptation.

### 2. Apply Domain Adaptation using D-MMD
To perform the adaptation, do:

```
    D-MMD.py
```

You can set the transfer problem you want by changing:

```python
	source = 'market1501'
	target = 'dukemtmcreid'
```