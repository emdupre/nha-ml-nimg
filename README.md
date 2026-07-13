# Learning with Nilearn: Statistical Learning, Machine Learning, and Alignment Applications in fMRI 

Materials presented at [NeuroHackademy 2026](https://neurohackademy.org) to introduce the software `nilearn` for statistical and machine learning analysis of neuroimaging data and `fmralign` for functional alignment of fMRI data.

To build the book locally, make sure that you have the requirements installed :

```
pip install -r requirements.txt
```

Note that this site is still built with jupyter-book 1.0 !

Then :

```
jupyter-book build content/
```

Generated notebooks are then available in `content/_build/jupyter_execute`,
and the rendered html is in  `content/_build/html/index.html`.