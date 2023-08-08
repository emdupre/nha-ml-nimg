---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  display_name: Python 3
  name: python3
repository:
  url: https://github.com/emdupre/nha-ml-nimg
---

# The importance of appropriate cross-validation


In using machine learning on neuroimaging data, appropriate cross-validation methods are critical for drawing meaningful inferences.
However, a majority of neuroscience researchers are not familiar with how to choose an appropriate method for their data.

```{figure} ../images/poldrack-2020-fig3.jpg
---
height: 250px
name: cv-usage
---
Figure 3 from {cite}`Poldrack_2020`, depicting esults from a review of 100 Studies (2017–2019) claiming prediction on fMRI Data
Panel A shows prevalence of cross-validation methods used to assess predictive accuracy in this sample.
Panel B shows a histogram of associated sample sizes.
```

First, we can formalize the problem that cross-validation is aimed at solving, adopting the notation used in {cite}`Little_2017`. 

For $N$ observations, we can choose a variable $y \in \mathbb{R}^n$ that we are trying to predict from data $X \in \mathbb{R}^{n \times p}$ in the presence of confounds $Z \in \mathbb{R}^{n \times k}$⁠.
For example, we may have neuroimaging data for 155 participants, from which we are trying to predict their age group as either a child or an adult.
There are additional confounding measures in this prediction, both measured and unmeasured.
For example, motion is a likely confounding variable, as children often move more in the scanner than adults.

In this notation, we can then consider $y$ as a function of X and Z:

$$
  y = f(XZ) + \epsilon
$$

If we assume strictly linear associations, we can re-write this function as a linear combination:

$$
  y = Xw + Zu + \epsilon
$$

where $\epsilon$ is observation noise.

In such model, $\epsilon$ may be i.i.d. even though the relationship between $y$ and $X$ is not i.i.d; for example, if it changes from subject to subject.

The machine learning problem is to estimate from train data {train} = (ytrainXtrain) a function $f̂{train}$ that predicts best $y$ from $X$.
In other words, we want to minimize an error (y,f̂ (X))⁠.
The purpose of cross-validation is to estimate this error. 

The challenge is that we are interested in the error on new, unknown, data, i.e. the expectancy of the error for $(y, X)$ drawn from their unknown distribution: 𝔼(y,X)[(y,f̂ (X))].
This is why evaluation procedures must test predictions of the model on left-out data that should be independent from the data used to train the model.

```{figure} ../images/varoquaux-2016-fig6.png
---
height: 350px
name: cv-strategies
---
Figure 6 from {cite}`Varoquaux_2016`.
This figures shows the difference in accuracy measured by cross-validation and on the held-out
validation set, in intra and inter-subject settings, for different cross-validation strategies:
- leave one sample out,
- leave one block of samples out (where the block is the natural unit of the experiment: subject or
session)
- random splits leaving out 20% of the blocks as test data, with 3, 10, or 50 random splits. 
For inter-subject settings, leave one sample out corresponds to leaving a session out.
The box gives the quartiles, while the whiskers give the 5 and 95 percentiles.
```

```{code-cell} python3
:tags: [hide-cell]

import warnings
warnings.filterwarnings("ignore")
```
In {ref}`an-example-classification-problem`, we used `StratifiedShuffleSplit` for cross-validation.
This method preserves the percentage of samples for each class across train and test splits; that is, the percentages of child and adult participants in our classification example.

Now that we've seen how to create a connectome for an individual subject,
we're ready to think about how we can use this connectome in a machine learning analysis.
We'll keep working with the same `development_dataset`,
but now we'd like to see if we can predict age group
(i.e. whether a participant is a child or adult) based on their connectome,
as defined by the functional connectivity matrix.

We'll also explore whether we're more or less accurate in our predictions based on how we define functional connectivity.
In this example, we'll consider three different different ways to define functional connectivity
between our Multi-Subject Dictional Learning (MSDL) regions of interest (ROIs):
correlation, partial correlation, and tangent space embedding.

To learn more about tangent space embedding and how it compares to standard correlations,
we recommend {cite}`Dadi_2019`.

## Load brain development fMRI dataset and MSDL atlas

First, we need to set up our minimal environment.
This will include all the dependencies from the last notebook,
loading the relevant data using our `nilearn` data set fetchers,
and instantiated our `NiftiMapsMasker` and `ConnectivityMeasure` objects.

```{code-cell} python3
:tags: [hide-output]
import numpy as np
import matplotlib.pyplot as plt
from nilearn import (datasets, maskers, plotting)
from nilearn.connectome import ConnectivityMeasure

development_dataset = datasets.fetch_development_fmri(n_subjects=30)
msdl_atlas = datasets.fetch_atlas_msdl()

masker = maskers.NiftiMapsMasker(
    msdl_atlas.maps, resampling_target="data",
    t_r=2, detrend=True,
    low_pass=0.1, high_pass=0.01).fit()
correlation_measure = ConnectivityMeasure(kind='correlation')
```

```{bibliography} references.bib
:style: unsrt
```
