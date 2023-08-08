---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
repository:
  url: https://github.com/emdupre/nha-ml-nimg
---

# The importance of cross-validation

```{code-cell} python3
:tags: [hide-cell]

import warnings
warnings.filterwarnings("ignore")
```

In applying machine learning to neuroimaging data, appropriate cross-validation methods are critical for drawing meaningful inferences.
However, many neuroscience researchers are not familiar with how to choose an appropriate cross-validation method for their data.

```{figure} ../images/poldrack-2020-fig3.jpg
---
height: 250px
name: cv-usage
---
From {cite}`Poldrack_2020`, depicting results from a review of 100 Studies (2017–2019) claiming prediction on fMRI data.
_Panel A_ shows the prevalence of cross-validation methods in this sample.
_Panel B_ shows a histogram of associated sample sizes.
```

We briefly overview what cross-validation aims to achieve, as well as several different strategies for cross-validation that are in use with neuroimaging data.
We then provide examples of appropriate and inappropriate cross-validation within the `development_fmri` dataset. 

## Why cross-validate ?

First, let's formalize the problem that cross-validation aims to solve, using notation from {cite}`Little_2017`. 

For $N$ observations, we can choose a variable $y \in \mathbb{R}^n$ that we are trying to predict from data $X \in \mathbb{R}^{n \times p}$ in the presence of confounds $Z \in \mathbb{R}^{n \times k}$⁠.
For example, we may have neuroimaging data for 155 participants, from which we are trying to predict their age group as either a child or an adult.
There are additional confounding measures in this prediction, both measured and unmeasured.
For example, motion is a likely confounding variable, as children often move more in the scanner than adults.

In this notation, we can then consider $y$ as a function of X and Z:

$$
  y = Xw + Zu + \epsilon
$$

where $\epsilon$ is observation noise, and we have assumed a strictly linear relationship between the variables.

In such model, $\epsilon$ may be independent and identically distributed (i.i.d.) even though the relationship between $y$ and $X$ is not i.i.d; for example, if it changes with age group membership.

The machine learning problem is to estimate a function $\hat{f}_{\{ train \}}$ that predicts best $y$ from $X$.
In other words, we want to minimize an error $\mathcal{E}(y,\hat{f}(X))$⁠.

The challenge is that we are interested in this error on new, unknown, data.
Thus, we would like to know the expectaction of the error for $(y, X)$ drawn from their unknown distribution:

$$
  \mathbb{E}_{(y,X)} [\mathcal{E}(y,\hat{f}(X))].
$$

From this we note two important points.
  1. Evaluation procedures _must_ test predictions of the model on held-out data that is independent from the data used to train the model.
  2. Cross-validation procedures that repeating the train-test split many times to vary the training set also allow use to ask a related question:
    given _future_ data to train a machine learning method on a clinical problem, what is the error that I can expect on new data?


## Forms of cross-validation

Given the importance of cross-validation in machine learning, many different general schemes exist.
The [scikit-learn documentation has a section](https://scikit-learn.org/stable/modules/cross_validation.html) just on this topic, which is worth reviewing in full.
Here, we briefly highlight several of the cross-validation methods already in use in neuroimaging.

## Leave-one-out can give overly optimistic estimates

In {cite}`Varoquaux_2017`, Varoquaux and colleagues evaluated the impact of different cross-validation schemes on derived accuracy values.
We reproduce their Figure 6 below.

```{figure} ../images/varoquaux-2016-fig6.png
---
height: 400px
name: cv-strategies
---
From {cite}`Varoquaux_2017` shows the difference in accuracy measured by cross-validation and on the held-out
validation set, in intra and inter-subject settings, for different cross-validation strategies:
(1) leave one sample out, (2) leave one block of samples out (where the block is the natural unit of the experiment: subject or session), and random splits leaving out 20% of the blocks as test data, with (3) 3, (4) 10, or (5) 50 random splits. 
For inter-subject settings, leave one sample out corresponds to leaving a session out.
The box gives the quartiles, while the whiskers give the 5 and 95 percentiles.
```

We see that cross-validation schemes that "leak" information from the train to test set can give overly optimistic predictions.
For example, if we leave-one-session-out for predictions within a participant, we see that our estimated prediction accuracy from cross-validation is much higher than our prediction accuracy on a held-out validation set.
This is because different sessions from the same participant are highly-correlated;
that is, participants are likely to show similar patterns of neural responses across sessions.

## Small sample sizes give a wide distribution of errors

Another common issue in leave-one-out cross-validation is the small size of the resulting test set.

```{figure} ../images/varoquaux-2017-fig1.png
---
height: 400px
name: test-size
---
From {cite}`Varoquaux_2018`, this plot shows the distribution of errors between the prediction accuracy as assessed via cross-validation (average across folds) and as measured on a large independent test set for different types of neuroimaging data.
Accuracy is reported for two reasonable choices of cross-validation strategy: leave-one-out (leave-one-run-out or leave-one-subject-out in data with multiple runs or subjects), or 50-times repeated splitting of 20% of the data.
The bar and whiskers indicate the median and the 5th and 95th percentile. 
```

The results show that these confidence bounds extends at least 10% both ways;
that is, there is a 5% chance that it is 10% above the true generalization accuracy and a 5% chance this it is 10% below.
This wide confidence bound is a result of an interaction between (1) the large sampling noise in neuroimaging data and (2) the relatively small sample sizes that we provide to the classifier.

## Testing cross-validation schemes in our example dataset.

We'll keep working with the same `development_dataset`, though this time we'll fetch all 155 subjects.
Again, we'll derive functional connectivity matrices for each participant. 

```{code-cell} python3
:tags: [hide-output]
import numpy as np
import matplotlib.pyplot as plt
from nilearn import (datasets, maskers, plotting)
from nilearn.connectome import ConnectivityMeasure

development_dataset = datasets.fetch_development_fmri()
msdl_atlas = datasets.fetch_atlas_msdl()

masker = maskers.NiftiMapsMasker(
    msdl_atlas.maps, resampling_target="data",
    t_r=2, detrend=True,
    low_pass=0.1, high_pass=0.01).fit()
correlation_measure = ConnectivityMeasure(kind='correlation')
```

In [our classification example](class-example), we used `StratifiedShuffleSplit` for cross-validation.
This method preserves the percentage of samples for each class across train and test splits; that is, the percentages of child and adult participants in our classification example.

```{code-call} python3
func_file = developmental_dataset.func[0]  # take the first subject functional
confound_file = developmental_datasets.confounds[0]  # and confounds file

time_series = masker.transform(func_file, confounds=confound_file)
correlation_matrices = correlation_measure.fit_transform(time_series)
```

```{bibliography} references.bib
:style: unsrt
:filter: docname in docnames
```
