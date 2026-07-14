---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.15.0
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Generalizability across subjects : Functional alignment with `fmralign`

```{code-cell} ipython3
:tags: [hide-cell]
import warnings
warnings.filterwarnings("ignore")
```

```{code-cell} ipython3
:tags: [hide-cell]
import os

os.environ["NILEARN_SHARED_DATA"] = "~/shared/data/nilearn_data"
```

Given that we have been largely working in a predictive context, we may be interested in looking at improving the generalizability of our results, rather than just assessing them.
In neuroimaging data, _functional alignment_ is a family of techniques which aim to do exactly this by increasing the similarity of subject-level functional data.
Starting with the introduction of _hyperalignment_ {cite}`10.7554/eLife.56601`, functional alignment has evolved to include a range of methods such as Procrustes, Optimal Transport, and the Shared Response Model.

In this example we will showcase a few of these methods using the Individual Brain Charting (IBC) {cite}`Pinho2018-dm` dataset, which include a large number of different contrasts maps for 12 subjects. 
This dataset includes contrasts derived from two independent sessions, each with a different phase encoding: Antero-Posterior(AP) or Postero-Anterior (PA).
We download the images for six of the subjects:

- `imgs` is the list of paths for each subjects
- `df` is a dataframe with metadata about each of subject
- `mask` is agroup-level brain mask for IBC data

```{code-cell} ipython3
from fmralign.fetch_example_data import fetch_ibc_subjects_contrasts

subjects = ["sub-01", "sub-02", "sub-04", "sub-05", "sub-06", "sub-07"]
imgs, df, mask_img = fetch_ibc_subjects_contrasts(subjects)
```
First, let's define a region of interest ; we will use the visual network of Yeo 2011 {cite}`Yeo2011-pl`.

```{code-cell} ipython3
from nilearn import datasets
from nilearn.image import concat_imgs, load_img, new_img_like, resample_to_img
from nilearn.plotting import plot_roi

atlas_yeo_2011 = datasets.fetch_atlas_yeo_2011()
atlas = load_img(atlas_yeo_2011.maps)

# Select visual cortex, create a mask and resample it to the right resolution

mask_visual = new_img_like(atlas, atlas.get_fdata() == 1)
resampled_mask_visual = resample_to_img(
    mask_visual, mask, interpolation="nearest"
)

# Plot the mask we will use
plot_roi(
    resampled_mask_visual,
    title="Visual regions mask extracted from atlas",
    cut_coords=(8, -80, 9),
    colorbar=True,
    cmap="Paired",
)
```

Next, we define a masker.

```{code-cell} ipython3
from nilearn.maskers import MultiNiftiMasker

roi_masker = MultiNiftiMasker(mask_img=resampled_mask_visual).fit()
```

Using our two different phase encoding directions, we can separate out a "train" and a "test" dataset.
Within each, we will take two subjects who we aim to make more similar.
The first will be the "source" subject and the second will be the "target" subject.

```{code-cell} ipython3
# The training fold, used to learn alignment from source subject toward target:
# * source train: AP contrasts for subject sub-01
# * target train: AP contrasts for subject sub-02

source_train_imgs = concat_imgs(
    df[(df.subject == "sub-01") & (df.acquisition == "ap")].path.values
)
target_train_imgs = concat_imgs(
    df[(df.subject == "sub-02") & (df.acquisition == "ap")].path.values
)

# The testing fold:
# * source test: PA contrasts for subject sub-01, used to predict
#   the corresponding contrasts of subject sub-02
# * target test: PA contrasts for subject sub-02, used as a ground truth
#   to score our predictions

source_test_imgs = concat_imgs(
    df[(df.subject == "sub-01") & (df.acquisition == "pa")].path.values
)
target_test_imgs = concat_imgs(
    df[(df.subject == "sub-02") & (df.acquisition == "pa")].path.values
)
```

Our goal will be to learn a transformation between the source and target subject,
with which we aim to improve their similarity in both the training and the testing datasets.


```{bibliography} references.bib
:style: unsrt
:filter: docname in docnames
```
