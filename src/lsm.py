"""
Latent space models and their generalizations.

Scott Linderman
2017
"""
import os, pickle

import numpy as np
import numpy.random as npr
from scipy.misc import logsumexp
npr.seed(0)

import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import seaborn as sns
sns.set_style("white")
sns.set_context("talk")

from pypolyagamma import pgdrawvpar, get_omp_num_threads, PyPolyaGamma
from pypolyagamma.utils import sample_gaussian


# Utils
onehot = lambda x, K: np.arange(K) == x
logistic = lambda x: 1 / (1 + np.exp(-x))


# Models
class LatentSpaceModel(object):
    def __init__(self, V, K, X=None, b=None, sigmasq_b=1.0):
        self.V, self.K = V, K

        # Initialize parameters
        self.X = npr.randn(V, K) if X is None else X * np.ones((V, K))
        self.b = np.zeros((V, V)) if b is None else b * np.ones((V, V))
        self.sigmasq_b = sigmasq_b

        # Models encapsulate data
        # A:  observed adjacency matrix
        # m:  mask for network n specifying which features to use
        # mask: mask specifying which entries in A were observed/hidden
        self.As = []
        self.ms = []
        self.masks = []

        # Polya-gamma RNGs
        num_threads = get_omp_num_threads()
        seeds = npr.randint(2 ** 16, size=num_threads)
        self.ppgs = [PyPolyaGamma(seed) for seed in seeds]

    def add_data(self, A, m=None, mask=None):
        V, D = self.V, self.K
        assert A.shape == (V, V) and A.dtype == bool
        self.As.append(A)

        m = m if m is not None else np.ones((D,))
        assert m.shape == (D,)
        self.ms.append(m)

        mask = mask if mask is not None else np.ones((V, V), dtype=bool)
        assert mask.shape == (V, V) and mask.dtype == bool
        self.masks.append(mask)

    def edge_probabilities(self, n):
        p = logistic((self.X * self.ms[n]).dot(self.X.T) + self.b)
        np.fill_diagonal(p, 0)
        return p

    def log_likelihood(self):
        return sum([self._log_likelihood(A, m, mask) for A, m, mask
                   in zip(self.As, self.ms, self.masks)])

    def heldout_log_likelihood(self):
        return sum([self._log_likelihood(A, m, ~mask) for A, m, mask
                    in zip(self.As, self.ms, self.masks)])

    def _log_likelihood(self, A, m, mask):
        V, X, b = self.V, self.X, self.b
        psi = self.b + (X * m).dot(X.T)

        L = np.tril(np.ones((V, V), dtype=bool), k=-1)
        ll = np.sum(psi * A * L * mask)
        ll -= np.sum(np.log1p(np.exp(psi)) * L * mask)
        return ll

    def generate(self, keep=True, mask=None):
        V, X, b = self.V, self.X, self.b
        m = np.ones(self.K)
        A = npr.rand(V, V) < logistic((X * m).dot(X.T) + b)

        # Make symmetric and remove self edges
        A[np.arange(V), np.arange(V)] = 0
        for u in range(V):
            for v in range(u):
                A[u, v] = A[v, u]

        if keep:
            self.add_data(A, mask=mask)

        return A

    ### Gibbs sampling
    def resample(self):
        self._resample_X()
        self._resample_b()

    def _resample_X(self):
        if self.K == 0:
            return

        # resample rows one at a time given the remainder
        for v in range(self.V):
            omegavs = self._resample_omegav(v)
            self._resample_xv(v, omegavs)

    def _resample_omegav(self, v):
        V = self.V
        notv = np.ones(V, dtype=bool)
        notv[v] = False
        omegas = []
        for n, (A, m) in enumerate(zip(self.As, self.ms)):
            # Scale the covariates by the mask
            Xnotv = self.X[notv] * m
            xv = self.X[v]
            psi = self.b[v][notv] + Xnotv.dot(xv)

            bb = np.ones(V-1)
            omega = np.zeros(V-1)
            pgdrawvpar(self.ppgs, bb, psi, omega)
            omegas.append(omega)
        return omegas

    def _resample_xv(self, v, omegavs):
        # Resample xv given {A_{n,v,:}, m_n, omega_{n,v,:}}_{n=1}^N and data masks
        V, K = self.V, self.K
        prior_J = np.eye(self.K)
        prior_h = np.zeros(self.K)

        lkhd_h = np.zeros(K)
        lkhd_J = np.zeros((K, K))

        notv = np.ones(V, dtype=bool)
        notv[v] = False
        for n, (A, m, mask, omega) in enumerate(zip(self.As, self.ms, self.masks, omegavs)):
            Xn = self.X[notv] * m
            Jn = omega * mask[v][notv]
            hn = (A[v][notv] - 0.5 - omega * self.b[v][notv]) * mask[v][notv]

            lkhd_J += (Xn * Jn[:, None]).T.dot(Xn)
            lkhd_h += hn.T.dot(Xn)

        post_h = prior_h + lkhd_h
        post_J = prior_J + lkhd_J
        self.X[v] = sample_gaussian(J=post_J, h=post_h)

    def _resample_b(self):
        V = self.V

        # Sample auxiliary variables
        XXTs = [(self.X * m).dot(self.X.T) for m in self.ms]
        psis = [self.b + XXT for XXT in XXTs]
        omegas = []
        for psi in psis:
            omega = np.zeros(V**2)
            pgdrawvpar(self.ppgs, np.ones(V**2), psi.ravel(), omega)
            omegas.append(omega.reshape((V, V)))

        # Sample b
        J = 1.0 / (self.sigmasq_b + 1e-8)
        h = 0.0
        for A, mask, XXT, omega in zip(self.As, self.masks, XXTs, omegas):
            J += omega * mask
            h += (A - 0.5 - omega * XXT) * mask

        sigmasq = 1. / J
        mu = sigmasq * h
        self.b = mu + np.sqrt(sigmasq) * npr.randn(V, V)

        # Symmetrize
        L = np.tril(np.ones((V, V)), k=-1)
        self.b = self.b * L + self.b.T * L.T


class MixtureOfLatentSpaceModels(LatentSpaceModel):
    """
    Extend the simple latent space model with a mixture distribution.
    This can be seen as setting a mask on some of the latent components.
    """

    def __init__(self, V, K, H, X=None, b=None, sigmasq_b=1.0, nu=None, alpha=1.0):
        """
        Here, K is the total number of factors (D * H)
        """
        assert K % H == 0, "K must be an integer multiple of H"
        self.D = K // H

        if X is not None:
            assert isinstance(X, list) and len(X) == H and \
                   np.all([Xh.shape == (V, K) for Xh in X])
            X = np.hstack(X)

        super(MixtureOfLatentSpaceModels, self).__init__(V, K, X=X, b=b, sigmasq_b=sigmasq_b)
        self.H = H
        self.hs = []

        # Prior on classes
        self.alpha = alpha
        self.nu = nu * np.ones(self.H) if nu is not None else alpha * np.ones(self.H)
        self.nu /= np.sum(self.nu)
        assert self.nu.shape == (self.H,) and np.all(self.nu >= 0)

    def add_data(self, A, m=None, mask=None, h=None):
        h = h if h is not None else npr.randint(self.H)
        assert isinstance(h, int) and h >= 0 and h < self.H
        self.hs.append(h)

        assert m is None
        m = np.repeat(onehot(h, self.H), self.D)
        super(MixtureOfLatentSpaceModels, self).add_data(A, m=m, mask=mask)

    def generate(self, keep=True, mask=None):
        V, X, b = self.V, self.X, self.b
        h = npr.randint(self.H)
        m = np.repeat(onehot(h, self.H), self.D)
        A = npr.rand(V, V) < logistic((X * m).dot(X.T) + b)

        # Make symmetric and remove self edges
        A[np.arange(V), np.arange(V)] = 0
        for u in range(V):
            for v in range(u):
                A[u, v] = A[v, u]

        if keep:
            self.add_data(A, mask=mask)

        return A

    def resample(self):
        self._resample_X()
        self._resample_b()
        self._resample_m()
        self._resample_nu()

    def _resample_m(self):
        for n, (A, mask) in enumerate(zip(self.As, self.masks)):
            lls = np.log(self.nu)
            for h in range(self.H):
                m = np.repeat(onehot(h, self.H), self.D)
                lls[h] += self._log_likelihood(A, m, mask)

            hn = npr.choice(self.H, p=np.exp(lls - logsumexp(lls)))
            self.hs[n] = hn
            self.ms[n] = np.repeat(onehot(hn, self.H), self.D)

    def _resample_nu(self):
        alpha_post = self.alpha / K + np.bincount(self.hs, minlength=self.H)
        self.nu = npr.dirichlet(alpha_post)


class FactorialLatentSpaceModel(LatentSpaceModel):
    def __init__(self, V, K, X=None, b=None, sigmasq_b=1.0, rho=None, alpha=1.0):
        super(FactorialLatentSpaceModel, self).__init__(V, K, X=X, b=b, sigmasq_b=sigmasq_b)

        # Prior on factors
        self.alpha = alpha
        self.rho = rho * np.ones(K) if rho is not None else npr.beta(alpha / K, 1.0, size=K)
        assert self.rho.shape == (self.K,) and np.all(self.rho >= 0)

    def add_data(self, A, m=None, mask=None):
        m = npr.rand(self.K) < self.rho
        super(FactorialLatentSpaceModel, self).add_data(A, m=m, mask=mask)

    def generate(self, keep=True, mask=None):
        V, X, b = self.V, self.X, self.b
        m = npr.rand(self.K) < self.rho
        A = npr.rand(V, V) < logistic((X * m).dot(X.T) + b)

        # Make symmetric and remove self edges
        A[np.arange(V), np.arange(V)] = 0
        for u in range(V):
            for v in range(u):
                A[u, v] = A[v, u]

        if keep:
            self.add_data(A, mask=mask)

        return A

    def resample(self):
        self._resample_X()
        self._resample_b()
        self._resample_m()
        self._resample_rho()

    def _resample_m(self):
        for n, (A, mask) in enumerate(zip(self.As, self.masks)):
            mn = self.ms[n].copy()
            for k in range(self.K):
                lls = np.array([np.log(1 - self.rho[k]), np.log(self.rho[k])])

                for s in range(2):
                    mn[k] = bool(s)
                    lls[s] += self._log_likelihood(A, mn, mask)

                # Sample
                mn[k] = np.log(npr.rand()) < lls[1] - logsumexp(lls)

            self.ms[n] = mn

    def _resample_rho(self):
        M = np.array(self.ms)
        alpha_post = self.alpha / self.K + M.sum(axis=0)
        beta_post = 1.0 + (1 - M).sum(axis=0)
        self.rho = npr.beta(alpha_post, beta_post)


def random_mask(V, missing_frac=0.1):
    mask = npr.rand(V, V) < 1 - missing_frac
    L = np.tril(np.ones((V, V), dtype=bool), k=-1)
    mask = mask * L + mask.T * L.T
    return mask


def synthetic_demo():
    V = 20                  # Number of vertices
    K = 2                   # Number of latent factors
    N = 50                  # Number of networks in population
    missing_frac = 0.1      # Fraction of data to withhold for testing
    N_itr = 20              # Number of iterations of sampler
    sigmasq_b = 0.0         # Prior variance of b (0 -> deterministically zero bias)

    # Sample data from a model with simple 2D covariates
    X = np.column_stack((np.linspace(-2, 2, V), np.zeros((V, K - 1))))
    true_model = LatentSpaceModel(V, K, X=X, sigmasq_b=sigmasq_b)
    masks = [random_mask(V, missing_frac) for _ in range(N)]
    As = [true_model.generate(keep=True, mask=mask) for mask in masks]
    true_ll = true_model.log_likelihood()
    true_hll = true_model.heldout_log_likelihood()

    def fit(model):
        print("Fitting ", model)
        lls = []
        hlls = []
        ms = []
        for itr in range(N_itr):
            if itr % 10 == 0:
                print("Iteration ", itr, " / ", N_itr)
            model.resample()
            lls.append(model.log_likelihood())
            hlls.append(model.heldout_log_likelihood())
            ms.append(model.ms)
        return np.array(lls), np.array(hlls), np.array(ms)

    # Fit the data with a standard LSM
    standard_lsm = LatentSpaceModel(V, K, sigmasq_b=sigmasq_b)
    for A, mask in zip(As, masks):
        standard_lsm.add_data(A, mask=mask)
    standard_lsm_lls, standard_lsm_hlls, standard_lsm_ms = fit(standard_lsm)

    # Fit the data with a mixture of LSMs
    mixture_lsm = MixtureOfLatentSpaceModels(V, 2*K, H=2, sigmasq_b=sigmasq_b)
    for A, mask in zip(As, masks):
        mixture_lsm.add_data(A, mask=mask)
    mixture_lsm_lls, mixture_lsm_hlls, mixture_lsm_ms = fit(mixture_lsm)

    # Fit the data with a factorial LSM
    factorial_lsm = FactorialLatentSpaceModel(V, K, sigmasq_b=sigmasq_b)
    for A, mask in zip(As, masks):
        factorial_lsm.add_data(A, mask=mask)
    factorial_lsm_lls, factorial_lsm_hlls, factorial_lsm_ms = fit(factorial_lsm)

    # Plot the results
    plt.figure(figsize=(12, 4))
    plt.subplot(141)
    plt.imshow(true_model.edge_probabilities(0), vmin=0, vmax=1, interpolation="nearest")
    plt.title("True Edge Probabilities")
    plt.subplot(142)
    plt.imshow(standard_lsm.edge_probabilities(0), vmin=0, vmax=1, interpolation="nearest")
    plt.title("Std LSM Edge Probabilities")
    plt.subplot(143)
    plt.imshow(mixture_lsm.edge_probabilities(0), vmin=0, vmax=1, interpolation="nearest")
    plt.title("Mixture LSM Edge Probabilities")
    plt.subplot(144)
    plt.imshow(factorial_lsm.edge_probabilities(0), vmin=0, vmax=1, interpolation="nearest")
    plt.title("Factorial LSM Edge Probabilities")

    plt.figure()
    plt.plot(standard_lsm_lls, label="Standard LSM")
    plt.plot(mixture_lsm_lls, label="Mixture of LSMs")
    plt.plot(factorial_lsm_lls, label="Factorial LSM")
    plt.plot([0, N_itr-1], true_ll * np.ones(2), ':k', label="True LSM")
    plt.xlabel("Iteration")
    plt.ylabel("Log Likelihood")
    plt.legend(loc="lower right")

    plt.figure()
    plt.plot(standard_lsm_hlls, label="Standard LSM")
    plt.plot(mixture_lsm_hlls, label="Mixture of LSMs")
    plt.plot(factorial_lsm_hlls, label="Factorial LSM")
    plt.plot([0, N_itr - 1], true_hll * np.ones(2), ':k', label="True LSM")
    plt.xlabel("Iteration")
    plt.ylabel("Heldout Log Likelihood")
    plt.legend(loc="lower right")
    plt.show()


if __name__ == "__main__":
    missing_frac = 0.25
    N_itr = 50
    H = 20
    K = 10
    sigmasq_b = 1.0

    # Load the KKI-42 dataset
    datapath = os.path.join("..", "data", "kki-42-data.pkl")
    assert os.path.exists(datapath)
    with open(datapath, "rb") as f:
        As = pickle.load(f)

    N, Vorig, _ = As.shape
    assert N == 42 and Vorig == 70 and As.shape[2] == Vorig
    bad_indices = [0, 35]
    good_indices = np.array(sorted(list(set(np.arange(Vorig)) - set(bad_indices))))
    As = As[np.ix_(np.arange(N), good_indices, good_indices)]
    V = Vorig - len(bad_indices)

    # Sample random masks
    masks = [random_mask(V, missing_frac) for _ in range(N)]

    def fit(model, name):
        print("Fitting ", name)
        lls = []
        hlls = []
        ms = []
        for itr in range(N_itr):
            if itr % 5 == 0:
                print("Iteration ", itr, " / ", N_itr)
            model.resample()
            lls.append(model.log_likelihood())
            hlls.append(model.heldout_log_likelihood())
            ms.append(model.ms)
        return np.array(lls), np.array(hlls), np.array(ms)

    Ks = [1, 2, 4, 6, 8, 10]
    experiments = []

    # Baseline: only include the bias term
    experiments.append((LatentSpaceModel(V, 0), "Bernoulli", "black"))

    # Standard Latent Space Models
    greens = get_cmap("Greens")
    for K in Ks:
        model = LatentSpaceModel(V, K, sigmasq_b=sigmasq_b)
        name = "standard_lsm_K{}".format(K)
        color = greens((1.0 + K) / 11.0)
        experiments.append((model, name, color))

    # Mixture of Latent Space Models
    reds = get_cmap("Reds")
    for K in Ks:
        model = MixtureOfLatentSpaceModels(V, K*H, H=H, sigmasq_b=sigmasq_b)
        name = "mixture_lsm_K{}_H{}".format(K, H)
        color = reds((1.0 + K) / 11.0)
        experiments.append((model, name, color))

    # Factorial Latent Space Models
    blues = get_cmap("Blues")
    for K in Ks:
        model = FactorialLatentSpaceModel(V, K, sigmasq_b=sigmasq_b, alpha=1 + K / 2.0)
        name = "factorial_lsm_K{}".format(K)
        color = blues((1.0 + K) / 11.0)
        experiments.append((model, name, color))

    # Fit the models
    results = []
    for model, name, _ in experiments:
        for A, mask in zip(As, masks):
            model.add_data(A, mask=mask)
        results.append(fit(model, name))

    # Plot the results
    plt.figure()
    for (model, name, color), result in zip(experiments, results):
        lls, hlls, ms = result
        plt.plot(lls, label=name, color=color)
    plt.xlabel("Iteration")
    plt.ylabel("Log Likelihood")
    plt.legend(loc="lower right")

    plt.figure()
    for (model, name, color), result in zip(experiments, results):
        lls, hlls, ms = result
        plt.plot(hlls, label=name, color=color)
    plt.xlabel("Iteration")
    plt.ylabel("Heldout Log Likelihood")
    plt.legend(loc="lower right")
    plt.show()

    # Now as a bar chart
    baseline_ll, baseline_hll, _ = results[0]
    baseline_ll = np.mean(baseline_ll[N_itr//2:])
    baseline_hll = np.mean(baseline_hll[N_itr//2:])

    plt.figure()
    for i, ((model, name, color), result) in enumerate(zip(experiments[1:], results[1:])):
        lls, hlls, ms = result
        plt.bar(i, np.mean(lls[N_itr // 2:]) - baseline_ll, color=color)
    plt.ylabel("Log Likelihood")
    plt.xticks(np.arange(len(experiments)-1), [e[1] for e in experiments[1:]], rotation=90)

    plt.figure()
    for i, ((model, name, color), result) in enumerate(zip(experiments[1:], results[1:])):
        lls, hlls, ms = result
        plt.bar(i, np.mean(hlls[N_itr//2:]) - baseline_hll, color=color)
    plt.ylabel("Heldout Log Likelihood")
    plt.xticks(np.arange(len(experiments) - 1), [e[1] for e in experiments[1:]], rotation=90)
    plt.show()


