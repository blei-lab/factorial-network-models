"""
Latent space models and a few generalizations.

Scott Linderman.  June 2017.
"""
import numpy as np
import numpy.random as npr
from scipy.misc import logsumexp
from scipy.stats import dirichlet, beta

from pypolyagamma import pgdrawvpar, get_omp_num_threads, PyPolyaGamma
from pypolyagamma.utils import sample_gaussian

from lsm.priors import SharedInvGammaPrior, MultiplicativeInvGammaPrior, MixtureofIGPriors, MixtureofMIGPriors
from lsm.utils import logistic, onehot


class LatentSpaceModel(object):

    _sigmasq_x_prior_class = SharedInvGammaPrior

    def __init__(self, V, K, X=None, b=None, sigmasq_b=1.0,
                 sigmasq_prior_prms=None, name=None):
        self.V, self.K = V, K

        # Initialize prior
        sigmasq_prior_prms = sigmasq_prior_prms if sigmasq_prior_prms is not None else {}
        self.sigmasq_x_prior = self._sigmasq_x_prior_class(K, **sigmasq_prior_prms)
        self.sigmasq_b = sigmasq_b

        # Initialize parameters
        self.X = np.sqrt(self.sigmasq_x) * npr.randn(V, K) if X is None else X * np.ones((V, K))

        self.b = np.zeros((V, V)) if b is None else b * np.ones((V, V))

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

        # Name the model
        self.name = name if name is not None else "lsm_K{}".format(K)

    @property
    def sigmasq_x(self):
        return self.sigmasq_x_prior.sigmasq

    def initialize(self):
        # Initialize latent variables if necessary
        pass

    def add_data(self, A, m=None, mask=None):
        V, D = self.V, self.K
        assert A.shape == (V, V) and A.dtype == bool
        self.As.append(A)

        m = m if m is not None else np.ones((D,))
        assert m.shape == (D,)
        self.ms.append(m)

        mask = mask if mask is not None else np.ones((V, V), dtype=bool)
        assert mask.shape == (V, V) and mask.dtype == bool
        assert np.all(mask.astype(float) - mask.astype(float).T == 0), "mask must be symmetric!"
        self.masks.append(mask)

    def edge_probabilities(self, n):
        p = logistic((self.X * self.ms[n]).dot(self.X.T) + self.b)
        np.fill_diagonal(p, 0)
        return p

    def log_prior(self):
        # p(X | sigmasq_x)
        lp = 0
        # if self.K > 0:
        #     sigmasq_x, X = self.sigmasq_x, self.X
        #     lp += -0.5 * np.sum(np.log(2 * np.pi * sigmasq_x)) * self.V
        #     lp += -0.5 * np.sum(X**2 / sigmasq_x)

        sigmasq_x, X = self.sigmasq_x, self.X
        for k in range(self.K):
            ssk = max(sigmasq_x[k], 1e-16)
            lp += -0.5 * np.sum(np.log(2 * np.pi * ssk)) * self.V
            lp += -0.5 * np.sum(X[:,k]**2 / ssk)

        # p(b | sigmasq_b)
        sigmasq_b, b = self.sigmasq_b, self.b
        L = np.tril(np.ones(b.shape), k=-1)
        lp += -0.5 * np.log(2 * np.pi * sigmasq_b) * L.sum()
        lp += -0.5 * np.sum(b**2 * L) / sigmasq_b

        # p(sigmasq_x)
        lp += self.sigmasq_x_prior.log_prior()
        return lp

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
        ll1 = np.sum(psi * A * L * mask)

        # Compute the log normalizer [-log(1+e^psi)] with logsumexp trick
        # This is equivalent to:
        # ll2 = np.sum(np.log1p(np.exp(psi)) * L * mask)
        lim = np.maximum(psi, 0)
        ll2 = np.sum((lim + np.log(np.exp(psi - lim) + np.exp(-lim))) * L * mask)
        # assert np.allclose(ll2, np.sum(np.log1p(np.exp(psi)) * L * mask))
        return ll1 - ll2

    def log_joint_probability(self):
        return self.log_prior() + self.log_likelihood()

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
        self.sigmasq_x_prior.resample(self.X)

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
        prior_J = np.diag(1. / self.sigmasq_x)
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

        # DEBUG: is the Cholesky in sample_gaussian more unstable than inv?
        # post_S = np.linalg.inv(post_J)
        # post_mu = np.dot(post_S, post_h)
        # self.X[v] = npr.multivariate_normal(post_mu, post_S)

    def _resample_b(self):
        V = self.V

        # Sample auxiliary variables.
        # We could be more efficient here since we only need
        # the lower triangular part.
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

        # Symmetrize -- only keep lower triangular part
        L = np.tril(np.ones((V, V)), k=-1)
        self.b = self.b * L + self.b.T * L.T


class LatentSpaceModelWithShrinkage(LatentSpaceModel):

    _sigmasq_x_prior_class = MultiplicativeInvGammaPrior

    def __init__(self, V, K, X=None, b=None, sigmasq_b=1.0,
                 sigmasq_prior_prms=None, name=None):
        super(LatentSpaceModelWithShrinkage, self).\
            __init__(V, K, X=X, b=b, sigmasq_b=sigmasq_b, sigmasq_prior_prms=sigmasq_prior_prms)
        self.name = name if name is not None else "lsm_shrink_K{}".format(K)


class MixtureOfLatentSpaceModels(LatentSpaceModel):
    """
    Extend the simple latent space model with a mixture distribution.
    This can be seen as setting a mask on some of the latent components.
    """
    _sigmasq_x_prior_class = MixtureofIGPriors

    def __init__(self, V, K, H, X=None, b=None, sigmasq_b=1.0,
                 sigmasq_prior_prms=None, name=None,
                 nu=None, alpha=1.0,):
        """
        Here, K is the total number of factors (D * H)
        """
        assert K % H == 0, "K must be an integer multiple of H"
        self.D = K // H

        # Make sure X is a list of correctly sized arrays
        if X is not None:
            assert isinstance(X, list) and len(X) == H and \
                   np.all([Xh.shape == (V, K) for Xh in X])
            X = np.hstack(X)

        sigmasq_prior_prms = sigmasq_prior_prms if sigmasq_prior_prms is not None else {}
        sigmasq_prior_prms["D"] = self.D
        super(MixtureOfLatentSpaceModels, self).\
            __init__(V, K, X=X, b=b, sigmasq_b=sigmasq_b,
                     sigmasq_prior_prms=sigmasq_prior_prms)

        # Initialize the cluster assignments
        self.H = H
        self.hs = []

        # Prior on classes
        self.alpha = alpha
        self.nu = nu * np.ones(self.H) if nu is not None else alpha * np.ones(self.H)
        self.nu /= np.sum(self.nu)
        assert self.nu.shape == (self.H,) and np.all(self.nu >= 0)

        self.name = name if name is not None else "molsm_K{}_H{}".format(K, H)

    def log_prior(self):
        lp = super(MixtureOfLatentSpaceModels, self).log_prior()
        # p({h} | nu)
        lp += np.dot(np.bincount(self.hs, minlength=self.H), np.log(1e-16 + self.nu))
        # p(nu)
        lp += dirichlet(self.alpha / self.H * np.ones(self.H)).logpdf(1e-16 + self.nu)
        return lp

    def initialize(self):
        # Cluster the adjacency matrices to initialize h
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=self.H)
        Aflat = np.array([A[np.tril_indices(self.V, k=-1)] for A in self.As])
        km.fit(Aflat)

        assert km.labels_.shape == (len(self.As),)
        for n, hn in enumerate(km.labels_):
            self.hs[n] = hn
            self.ms[n] = np.repeat(onehot(hn, self.H), self.D)
        assert np.all(np.sum(self.ms, 0) > 0)

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
        super(MixtureOfLatentSpaceModels, self).resample()
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
        alpha_post = self.alpha / self.K + np.bincount(self.hs, minlength=self.H)
        self.nu = npr.dirichlet(alpha_post)


class MixtureOfLatentSpaceModelsWithShrinkage(MixtureOfLatentSpaceModels):

    _sigmasq_x_prior_class = MixtureofMIGPriors

    def __init__(self, V, K, H, X=None, b=None, sigmasq_b=1.0, nu=None,
                 alpha=1.0, sigmasq_prior_prms=None, name=None):
        super(MixtureOfLatentSpaceModelsWithShrinkage, self).\
            __init__(V, K, H, X=X, b=b, sigmasq_b=sigmasq_b,
                     sigmasq_prior_prms=sigmasq_prior_prms,
                     nu=nu, alpha=alpha)
        self.name = name if name is not None else "molsm_shrink_K{}_H{}".format(K, H)


class FactorialLatentSpaceModel(LatentSpaceModel):
    def __init__(self, V, K, X=None, b=None, sigmasq_b=1.0,
                 sigmasq_prior_prms=None, name=None,
                 rho=None, alpha=1.0):
        super(FactorialLatentSpaceModel, self).\
            __init__(V, K, X=X, b=b, sigmasq_b=sigmasq_b, sigmasq_prior_prms=sigmasq_prior_prms)

        # Prior on factors
        self.alpha = alpha
        self.rho = rho * np.ones(K) if rho is not None else npr.beta(alpha / K, 1.0, size=K)
        assert self.rho.shape == (self.K,) and np.all(self.rho >= 0)

        self.name = name if name is not None else "flsm_K{}".format(K)

    def log_prior(self):
        lp = super(FactorialLatentSpaceModel, self).log_prior()

        # p(rho)
        for r in self.rho:
            lp += beta(self.alpha / self.K, 1.0).logpdf(r)

        # p({m} | rho)
        msum = np.sum(self.ms, axis=0)
        lp += np.dot(msum, np.log(self.rho))
        lp += np.dot(len(self.ms) - msum, np.log(1-self.rho))
        return lp

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
        super(FactorialLatentSpaceModel, self).resample()
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
