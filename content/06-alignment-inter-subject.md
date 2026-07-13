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
In neuroimaging data, _functional alignment_ is a family of techniques which aim to do exactly this.
Starting with the introduction of _hyperalignment_

```{code-cell} ipython3
:tags: [hide-cell]
:mystnb:
:  code_prompt_show: "Show code for dataset loading"
:  code_prompt_hide: "Hide code for dataset loading"
from fmralign.fetch_example_data import fetch_ibc_subjects_contrasts

subjects = ["sub-01", "sub-02", "sub-04", "sub-05", "sub-06", "sub-07"]
imgs, df, mask_img = fetch_ibc_subjects_contrasts(subjects)
```

```{bibliography} references.bib
:style: unsrt
:filter: docname in docnames
```
