PAC2019
=======

# Abstract
We ranked 3rd in the [PAC 2019 challenge](https://www.photon-ai.com/pac2019), by achieving a MAE of 3.33 years in predicting age from T1w MRI brain images. Our approach combined 7 algorithms that allow making predictions when the number of features exceed the number of observations. Namely, 2 versions of Best Linear Unbiased Predictor (BLUP),  Support Vector Machine (SVM), 2 shallow Convolutional Neural Networks (CNN), as well as the famous ResNet and Inception V1. Ensemble learning was derived from estimating weights via linear regression in an hold-out subset of the training sample. 
We further evaluated and identified factors that could influence prediction accuracy: choice of algorithm, ensemble learning, features used as input / MRI image processing. Our prediction error was correlated with age and absolute error was greater for older participants, suggesting to increase the training sample for this sub-group. 
Our results may be used to guide researchers to build age predictors on healthy individuals, that can be used in research and in the clinics as non-specific predictors of disease status.   

Keywords: Brain age, aging, MRI, machine learning, deep learning, statistical learning


# Contributors
A **huge** thank you to all the contributors to this challenge:
- [Baptiste Couvy-Duchesne](https://github.com/baptisteCD)
- [Johann Faouzi](https://github.com/johannfaouzi)
- [Benoit Martin](https://github.com/benoitmartin88)
- [Elina Thibeau-Sutre](https://github.com/14thibea)
- [Adam Wild](https://github.com/adamwild)
- [Manon Ansart](https://github.com/manonansart)


# Repository structure
This repository is structures as follows:

``` bash
.
├── README.md
├── data
├── md
├── requirements.txt
├── results_additional
├── results_main
├── scripts
├── src
└── statistics and visualization
```

From the repository's root, several folders are accessible. Namely, the `data` folder contains the training and validation splits that have been generated.
The `md` folder contains documentation in a markdown format.
The `results_main`, `results_additional` and `statistics and visualization` folders contain the results produced for the PAC challenge and a few additional experiments that were performed after the challenge's deadline.
All the scripts that have been used during the challenge can be found in the `scripts` folder.
The folder that possibly is of most interest if the results are to be reproduced is the `src` folder. The complete source code used for this challenge can be found in this folder.

Please note that this repository uses 2 a mix programing languages: **R** and **Python**.


# Requirements
- Python >= 3.6
- [R](https://www.r-project.org/)
- [Jupyter notebooks](https://jupyter.org/)
- [OSCA](http://cnsgenomics.com/software/osca/)
- [Freesurfer](https://surfer.nmr.mgh.harvard.edu/)
- [Pytorchtrainer](https://pypi.org/project/pytorchtrainer/)

The full python requirements are in the `requirements.txt` file at the repository root.

