---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  display_name: Python 3
  name: python3
repository:
  url: https://github.com/emdupre/nha2020-nilearn
---

# An introduction to [`nilearn`](http://nilearn.github.io)

```{code-cell} python3
:tags: [hide-cell]

import warnings
warnings.filterwarnings("ignore")
```

In this tutorial, we'll see how the Python library _nilearn_ allows us to easily perform machine learning analyses with neuroimaging data, specifically MRI and fMRI.

You may notice that the name `nilearn` is reminiscent of [`scikit-learn`](https://scikit-learn.org),
a popular Python library for machine learning.
This is no accident!
Nilearn and scikit-learn were created by the same team,
and nilearn is designed to bring machine **LEARN**ing to the NeuroImaging (**NI**) domain.

With that in mind, let's briefly consider why we might want specialized tools for working with neuroimaging data.
When performing a machine learning analysis, our data often look something like this:

NB: update to Tal's example once uploaded !

```{code-cell} python3
from sklearn import datasets

diabetes = datasets.load_diabetes(as_frame=True)
diabetes.data.head()
```

This data is clearly structured in a tabular format.
This makes it easier to consider issues such as: which features would we like to predict?
Or, how do we handle cross-validation?

## Neuroimaging data

By contrast, neuroimaging data has a very different structure.
Neuroimaging data has both spatial and temporal dependencies between successive data points.
That is, knowing _where_ and _when_ something was measured tells you information about the surrounding data points.
We also know that neuroimaging data contains a lot of noise that's not blood-oxygen-level dependent (BOLD), such as head motion.
Since we don't think that these other noise sources are related to neuronal firing,
we often need to consider how we can make sure that our analyses are not driven by these noise sources.
These are all considerations that most machine learning software libraries are not designed to deal with !
Nilearn therefore plays a crucial role in bringing machine learning concepts to the neuroimaging domain.

To get a sense of the problem, the quickest method is to just look at some neuroimaging data.
You may have your own data locally that you'd like to work with.
Nilearn also provides access to several neuroimaging data sets and atlases (we'll talk about these a bit later).
These data sets (and atlases) are only accessible because research groups chose to make their collected data publicly available.
We owe them a huge thank you for this !
The data set we'll use today was originally collected by [Rebecca Saxe](https://mcgovern.mit.edu/profile/rebecca-saxe/)'s group at MIT and hosted on [OpenNeuro](https://openneuro.org/datasets/ds000228/versions/1.1.0).

The nilearn team preprocessed the data set with [fMRIPrep](https://fmriprep.readthedocs.io) and downsampled it to a lower resolution,
so it'd be easier to work with.
We can learn a lot about this data set directly [from the Nilearn documentation](https://nilearn.github.io/modules/generated/nilearn.datasets.fetch_development_fmri.html)!
For example, we can see that this data set contains over 150 children and adults watching a short Pixar film.
Let's download the first 30 participants.

```{code-cell} python3
:tags: [hide-output]
from nilearn import datasets

development_dataset = datasets.fetch_development_fmri(n_subjects=30)
```

Now, this `development_dataset` object has several attributes which provide access to the relevant information.
For example, `development_dataset.phenotypic` provides access to information about the participants, such as whether they were children or adults.
We can use `development_dataset.func` to access the functional MRI (fMRI) data.

## Getting into the data: subsetting and viewing

Nilearn also provides many methods for plotting this kind of data.
For example, we can use [`nilearn.plotting.view_img`](https://nilearn.github.io/modules/generated/nilearn.plotting.view_img.html) to launch at interactive viewer.
Because each fMRI run is a 4D time series (three spatial dimensions plus time),
we'll also need to subset the data when we plot it, so that we can look at a single 3D image.
Nilearn provides (at least) two ways to do this: with [`nilearn.image.index_img`](https://nilearn.github.io/modules/generated/nilearn.image.index_img.html),
which allows us to index a particular frame--or several frames--of a time series,
and [`nilearn.image.mean_img`](https://nilearn.github.io/modules/generated/nilearn.image.mean_img.html),
which allows us to take the mean 3D image over time.

Putting these together, we can interatively view the mean image of the first participant using:

```{code-cell} python3
import matplotlib.pyplot as plt
from nilearn import image
from nilearn import plotting

mean_image = image.mean_img(development_dataset.func[0])
plotting.view_img(mean_image, threshold=None)
```

## Extracting signal from fMRI volumes

As you can see, this data is decidedly not tabular !
What we'd like is to extract and transform relevant features from this data into a tabular format that we can more easily work with.
To do this, we'll use nilearn's Masker objects.
What are the masker objects ?
First, let's think about what masking fMRI data is doing:

```{figure} images/masking.jpg
---
height: 350px
name: masking
---
Masking fMRI data.
```

Essentially, we can imagine overlaying a 3D grid on an image.
Then, our mask tells us which cubes or “voxels” (like 3D pixels) to sample from.
Since our Nifti images are 4D files, we can’t overlay a single grid –
instead, we use a series of 3D grids (one for each volume in the 4D file),
so we can get a measurement for each voxel at each timepoint.

Masker objects allow us to apply these masks !
To start, we need to define a mask (or masks) that we'd like to apply.
This could correspond to one or many regions of interest.
Nilearn provides methods to define your own functional parcellation (using clustering algorithms such as k-means),
and it also provides access to other atlases that have previously been defined by researchers.

## Choosing regions of interest

In this tutorial,
we'll use the MSDL (multi-subject dictionary learning; {cite}`Varoquaux_2011`) atlas,
which defines a set of _probabilistic_ regions of interest (ROIs) across the brain.

```{code-cell} python3
import numpy as np

msdl_atlas = datasets.fetch_atlas_msdl()

msdl_coords = msdl_atlas.region_coords
n_regions = len(msdl_coords)

print(f'MSDL has {n_regions} ROIs, part of the following networks :\n{np.unique(msdl_atlas.networks)}.')
```

Nilearn also provides us with an easy way to view this atlas directly:

```{code-cell} python3
plotting.plot_prob_atlas(msdl_atlas.maps)
```

We'd like to supply these ROIs to a Masker object.
All Masker objects share the same basic structure and functionality,
but each is designed to work with a different kind of ROI.

The canonical [`nilearn.input_data.NiftiMasker`](https://nilearn.github.io/modules/generated/nilearn.input_data.NiftiMasker.html) works well if we want to apply a single mask to the data,
like a single region of interest.

But what if we actually have several ROIs that we'd like to apply to the data all at once?
If these ROIs are non-overlapping,
as in "hard" or deterministic parcellations,
then we can use [`nilearn.input_data.NiftiLabelsMasker`](https://nilearn.github.io/modules/generated/nilearn.input_data.NiftiLabelsMasker.html).
Because we're working with "soft" or probabilistic ROIs,
we can instead supply these ROIs to [`nilearn.input_data.NiftiMapsMasker`](https://nilearn.github.io/modules/generated/nilearn.input_data.NiftiMapsMasker.html).

For a full list of the available Masker objects,
see [the Nilearn documentation](https://nilearn.github.io/modules/reference.html#module-nilearn.input_data).

We can supply our MSDL atlas-defined ROIs to the `NiftiMapsMasker` object,
along with resampling, filtering, and detrending parameters.

```{code-cell} python3
from nilearn import input_data

masker = input_data.NiftiMapsMasker(
    msdl_atlas.maps, resampling_target="data",
    t_r=2, detrend=True,
    low_pass=0.1, high_pass=0.01).fit()
```

One thing you might notice from the above code is that immediately after defining the masker object,
we call the `.fit` method on it.
This method may look familiar if you've previously worked with scikit-learn estimators!

```{code-cell} python3
```

They also allow us to perform other operations,
like correcting for measured signals of no interest (e.g., head motion).
Our `development_dataset` also includes several of these signals of no interest that were generated by fMRIPrep pre-processing.
We can access these with the `confounds` attribute,
using `development_dataset.confounds`.

Let's quickly check what these look like for our first participant:

```{code-cell} python3
import pandas as pd

pd.read_table(development_dataset.confounds[0]).head()
```

## An example classification problem

This example compares different kinds of functional connectivity between
regions of interest: correlation, partial correlation, and tangent space
embedding.

The resulting connectivity coefficients can be used to
discriminate children from adults. In general, the tangent space embedding
**outperforms** the standard correlations: see {cite}`Dadi_2019`
for a careful study.

## Load brain development fMRI dataset and MSDL atlas

We study only 30 subjects from the dataset, to save computation time.

```{code-cell} python3
:tags: [hide-output]
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.svm import LinearSVC

from nilearn.connectome import ConnectivityMeasure
```

## Region signals extraction

To extract regions time series, we instantiate a
:class:`nilearn.input_data.NiftiMapsMasker` object and pass the atlas the
file name to it, as well as filtering band-width and detrending option.

Then we compute region signals and extract useful phenotypic informations.

```{code-cell} python3
children = []
pooled_subjects = []
groups = []  # child or adult
for func_file, confound_file, phenotypic in zip(
        development_dataset.func,
        development_dataset.confounds,
        development_dataset.phenotypic):
    time_series = masker.transform(func_file, confounds=confound_file)
    pooled_subjects.append(time_series)
    if phenotypic['Child_Adult'] == 'child':
        children.append(time_series)
    groups.append(phenotypic['Child_Adult'])

print('Data has {0} children.'.format(len(children)))
```

## ROI-to-ROI correlations of children

The simpler and most commonly used kind of connectivity is correlation. It
models the full (marginal) connectivity between pairwise ROIs. We can
estimate it using :class:`nilearn.connectome.ConnectivityMeasure`.

```{code-cell} python3
correlation_measure = ConnectivityMeasure(kind='correlation')
```

From the list of ROIs time-series for children, the
`correlation_measure` computes individual correlation matrices.

```{code-cell} python3
correlation_matrices = correlation_measure.fit_transform(children)
```

All individual coefficients are stacked in a unique 2D matrix.

```{code-cell} python3
print('Correlations of children are stacked in an array of shape {0}'
      .format(correlation_matrices.shape))
```

as well as the average correlation across all fitted subjects.

```{code-cell} python3
mean_correlation_matrix = correlation_measure.mean_
print('Mean correlation has shape {0}.'.format(mean_correlation_matrix.shape))
```

We display the connectome matrices of the first 3 children

```{code-cell} python3
_, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, (matrix, ax) in enumerate(zip(correlation_matrices, axes)):
    plotting.plot_matrix(matrix, tri='lower', colorbar=False, axes=ax,
                         title='correlation, child {}'.format(i))
```

The blocks structure that reflect functional networks are visible.

Now we display as a connectome the mean correlation matrix over all children.

```{code-cell} python3
plotting.plot_connectome(mean_correlation_matrix, msdl_coords,
                         title='mean correlation over all children')
```

## Studying partial correlations

We can also study **direct connections**, revealed by partial correlation
coefficients. We just change the `ConnectivityMeasure` kind

```{code-cell} python3
partial_correlation_measure = ConnectivityMeasure(kind='partial correlation')
partial_correlation_matrices = partial_correlation_measure.fit_transform(
    children)
```

Most of direct connections are weaker than full connections.

```{code-cell} python3
_, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, (matrix, ax) in enumerate(zip(partial_correlation_matrices, axes)):
    plotting.plot_matrix(matrix, tri='lower', colorbar=False, axes=ax,
                         title='partial correlation, child {}'.format(i))
```

```{code-cell} python3
plotting.plot_connectome(
    partial_correlation_measure.mean_, msdl_coords,
    title='mean partial correlation over all children')
```

## Extract subjects variabilities around a group connectivity

We can use **both** correlations and partial correlations to capture
reproducible connectivity patterns at the group-level.
This is done by the tangent space embedding.

```{code-cell} python3
tangent_measure = ConnectivityMeasure(kind='tangent')
```

We fit our children group and get the group connectivity matrix stored as
in `tangent_measure.mean_`, and individual deviation matrices of each subject
from it.

```{code-cell} python3
tangent_matrices = tangent_measure.fit_transform(children)
```

`tangent_matrices` model individual connectivities as
**perturbations** of the group connectivity matrix `tangent_measure.mean_`.
Keep in mind that these subjects-to-group variability matrices do not
directly reflect individual brain connections. For instance negative
coefficients can not be interpreted as anticorrelated regions.

```{code-cell} python3
_, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, (matrix, ax) in enumerate(zip(tangent_matrices, axes)):
    plotting.plot_matrix(matrix, tri='lower', colorbar=False, axes=ax,
                         title='tangent offset, child {}'.format(i))
```

The average tangent matrix cannot be interpreted, as individual matrices
represent deviations from the mean, which is set to 0.

## What kind of connectivity is most powerful for classification?

We will use connectivity matrices as features to distinguish children from
adults. We use cross-validation and measure classification accuracy to
compare the different kinds of connectivity matrices.
We use random splits of the subjects into training/testing sets.
StratifiedShuffleSplit allows preserving the proportion of children in the
test set.

```{code-cell} python3
kinds = ['correlation', 'partial correlation', 'tangent']
_, classes = np.unique(groups, return_inverse=True)
cv = StratifiedShuffleSplit(n_splits=15, random_state=0, test_size=5)
pooled_subjects = np.asarray(pooled_subjects)

scores = {}
for kind in kinds:
    scores[kind] = []
    for train, test in cv.split(pooled_subjects, classes):
        # *ConnectivityMeasure* can output the estimated subjects coefficients
        # as a 1D arrays through the parameter *vectorize*.
        connectivity = ConnectivityMeasure(kind=kind, vectorize=True)
        # build vectorized connectomes for subjects in the train set
        connectomes = connectivity.fit_transform(pooled_subjects[train])
        # fit the classifier
        classifier = LinearSVC().fit(connectomes, classes[train])
        # make predictions for the left-out test subjects
        predictions = classifier.predict(
            connectivity.transform(pooled_subjects[test]))
        # store the accuracy for this cross-validation fold
        scores[kind].append(accuracy_score(classes[test], predictions))
```

display the results

```{code-cell} python3
mean_scores = [np.mean(scores[kind]) for kind in kinds]
scores_std = [np.std(scores[kind]) for kind in kinds]

plt.figure(figsize=(6, 4))
positions = np.arange(len(kinds)) * .1 + .1
plt.barh(positions, mean_scores, align='center', height=.05, xerr=scores_std)
yticks = [k.replace(' ', '\n') for k in kinds]
plt.yticks(positions, yticks)
plt.gca().grid(True)
plt.gca().set_axisbelow(True)
plt.gca().axvline(.8, color='red', linestyle='--')
plt.xlabel('Classification accuracy\n(red line = chance level)')
plt.tight_layout()
```

This is a small example to showcase nilearn features. In practice such
comparisons need to be performed on much larger cohorts and several
datasets.
{cite}`Dadi_2019` showed that across many cohorts and clinical questions,
the tangent kind should be preferred.

```{bibliography} references.bib
:style: unsrt
```
