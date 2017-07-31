# Factorial Network Models
A discussion of "Nonparametric Bayes Modeling of Populations of Networks" by Durante et al. (2016).

_Scott W. Linderman and David M. Blei_

### Abstract
While the modeling of _single_ networks has received much attention, we agree with Durante et al. that the modeling of _populations_ of networks has been largely overlooked. Given that such data are increasingly common in fields such as neuroscience, their paper makes a timely contribution. In this discussion, we consider a factorial generalization of the proposed mixture of latent space models, and we suggest cases in which factorial models may naturally capture our intuition about the underlying generative process of the data. We compare these two models using the human brain data studied by Durante et al., and we suggest some avenues for future work. 

### Organization
- `data`: helpers to download and extract the brain data.
- `doc`: LaTeX source of the discussion paper.
- `lsm`: source code for the latent space model (LSM), the mixture of latent space models (MoLSM), and the factorial latent space model (fLSM).
- `notebooks`: analysis of synthetic data and brain data. This includes code to reproduce the figures in the paper.
- `results`: directory where model fits will be cached.


### Installation
First, make sure you have [pypolyagamma](https://github.com/slinderman/pypolyagamma) installed.  This can be done with `pip` or by source. See the link above for detailed installation instructions.

Then,
```
git clone git@github.com:blei-lab/factorial-network-models.git
cd factorial-network-models
pip install -e .
```

