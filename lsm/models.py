"""
Latent space models and their generalizations.

Scott Linderman.  June 2017.
"""
import numpy as np
import numpy.random as npr
from scipy.misc import logsumexp

from pypolyagamma import pgdrawvpar, get_omp_num_threads, PyPolyaGamma
from pypolyagamma.utils import sample_gaussian

from lsm.utils import logistic, onehot


class LatentSpaceModel(object):
    def __init__(self, V, K, X=None, b=None, sigmasq_b=1.0, sigmasq_x=1.0):
        self.V, self.K = V, K

        # Initialize parameters
        self._sigmasq_x = sigmasq_x * np.ones(K)
        self.X = np.sqrt(self.sigmasq_x) * npr.randn(V, K) if X is None else X * np.ones((V, K))

        self.sigmasq_b = sigmasq_b
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

    @property
    def sigmasq_x(self):
        return self._sigmasq_x

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


class MultiplicativeInvGammaPrior(object):
    """
    Shrinkage prior on the scale of the latent embeddings
    """
    def __init__(self, K, a1, a2):
        self.K, self.a1, self.a2 = K, a1, a2
        self.nu = np.ones(K)

        # Resample from prior to initialize nu
        self.resample(np.zeros((0, K)))

    @property
    def sigmasq(self):
        return 1. / np.cumprod(self.nu)

    def resample(self, X):
        assert isinstance(X, np.ndarray) and X.ndim == 2 and X.shape[1] == self.K
        V, K = X.shape

        # Define a helper function to take product of certain fraction of nu's.
        def theta(m, k):
            th = 1.0
            for t in range(m):
                if t != k:
                    th *= self.nu[t]
            return th

        # Resample the nu_1; this one is special
        a1_post = self.a1 + 0.5 * V * K
        b1_post = 1.0
        for i in range(K):
            b1_post += 0.5 * theta(i, 1) * np.sum(X[:, i]**2)
        self.nu[0] = npr.gamma(a1_post, 1./b1_post)

        # Resample the nu_k; k >= 2
        for k in range(1, K):
            ak_post = self.a2 + 0.5 * V * (K - k)
            bk_post = 1.0
            for i in range(k, K):
                bk_post += 0.5 * theta(i, k) * np.sum(X[:, i] ** 2)
            self.nu[k] = npr.gamma(ak_post, 1. / bk_post)


class LatentSpaceModelWithShrinkage(LatentSpaceModel):
    def __init__(self, V, K, X=None, b=None, sigmasq_b=1.0, a1=1.0, a2=1.0):
        self.sigmasq_x_prior = MultiplicativeInvGammaPrior(K, a1, a2)
        super(LatentSpaceModelWithShrinkage, self).\
            __init__(V, K, X=X, b=b, sigmasq_b=sigmasq_b)

    @property
    def sigmasq_x(self):
        return self.sigmasq_x_prior.sigmasq

    def resample(self):
        super(LatentSpaceModelWithShrinkage, self).resample()
        self.sigmasq_x_prior.resample(self.X)


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
        alpha_post = self.alpha / self.K + np.bincount(self.hs, minlength=self.H)
        self.nu = npr.dirichlet(alpha_post)


class MixtureOfLatentSpaceModelsWithShrinkage(MixtureOfLatentSpaceModels):
    def __init__(self, V, K, H, X=None, b=None, sigmasq_b=1.0, nu=None, alpha=1.0, a1=1.0, a2=1.0):
        self.sigmasq_x_priors = [MultiplicativeInvGammaPrior(K // H, a1, a2) for _ in range(H)]

        super(MixtureOfLatentSpaceModelsWithShrinkage, self).\
            __init__(V, K, H, X=X, b=b, sigmasq_b=sigmasq_b, nu=nu, alpha=alpha)

    @property
    def sigmasq_x(self):
        return np.hstack([prior.sigmasq for prior in self.sigmasq_x_priors])

    def resample(self):
        super(MixtureOfLatentSpaceModelsWithShrinkage, self).resample()

        # Resample prior
        for h, prior in enumerate(self.sigmasq_x_priors):
            prior.resample(self.X[:, h*self.D:(h+1)*self.D])


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
