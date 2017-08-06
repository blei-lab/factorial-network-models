import numpy as np
import numpy.random as npr
from scipy.stats import gamma


class SharedInvGammaPrior(object):
    """
    Shared inverse gamma prior for all of X
    """
    def __init__(self, K, a=1.0, b=1.0):
        self.K = K

        assert a > 0 and b > 0, "shape and scale must be greater than 0"
        self.a, self.b = a, b
        self.nu = npr.gamma(a, 1. / b)

    @property
    def sigmasq(self):
        return np.ones(self.K) / self.nu

    def log_prior(self):
        return gamma(self.a, scale=1./self.b).logpdf(self.nu)

    def resample(self, X):
        assert isinstance(X, np.ndarray) and X.ndim == 2 and X.shape[1] == self.K
        V, K = X.shape

        # Resample the nu_1; this one is special
        a_post = self.a + 0.5 * V * K
        b_post = self.b + 0.5 * np.sum(X**2)
        self.nu = npr.gamma(a_post, 1. / b_post)


class MultiplicativeInvGammaPrior(object):
    """
    Shrinkage prior on the scale of the latent embeddings
    """
    def __init__(self, K, a1=2.5, a2=3.5):
        self.K, self.a1, self.a2 = K, a1, a2
        self.nu = np.ones(K)

        # Resample from prior to initialize nu
        self.resample(np.zeros((0, K)))

    @property
    def sigmasq(self):
        # Truncate for numerical stability
        return 1. / np.cumprod(self.nu)

    def log_prior(self):
        ll = gamma(self.a1, scale=1).logpdf(self.nu[0])
        ll += np.sum(gamma(self.a2, scale=1).logpdf(self.nu[1:]))
        return ll

    def resample(self, X):
        assert isinstance(X, np.ndarray) and X.ndim == 2 and X.shape[1] == self.K
        V, K = X.shape

        # Define a helper function to take product of certain fraction of nu's.
        def theta(m, r):
            th = 1.0
            for t in range(m+1):
                if t != r:
                    th *= self.nu[t]
            return th

        # Resample the nu_1; this one is special
        a1_post = self.a1 + 0.5 * V * K
        b1_post = 1.0
        for m in range(K):
            b1_post += 0.5 * theta(m, 0) * np.sum(X[:, m]**2)
        self.nu[0] = npr.gamma(a1_post, 1. / b1_post)

        # Resample the nu_k; k >= 2
        for k in range(1, K):
            ak_post = self.a2 + 0.5 * V * (K - k)
            bk_post = 1.0
            for m in range(k, K):
                bk_post += 0.5 * theta(m, k) * np.sum(X[:, m] ** 2)
            self.nu[k] = npr.gamma(ak_post, 1. / bk_post)


class _MixturePriorBase(object):

    _prior_class = None

    def __init__(self, K, D, **prms):
        self.K, self.D = K, D
        assert K % D == 0
        self.H = K // D
        self.priors = [self._prior_class(D, **prms) for _ in range(self.H)]

    @property
    def sigmasq(self):
        return np.hstack([prior.sigmasq for prior in self.priors])

    def log_prior(self):
        return sum([prior.log_prior() for prior in self.priors])

    def resample(self, X):
        H, D = self.H, self.D
        assert X.ndim == 2 and X.shape[1] == H * D
        for h in range(H):
            Xh = X[:, h*D:(h+1)*D]
            self.priors[h].resample(Xh)


class MixtureofIGPriors(_MixturePriorBase):
    _prior_class = SharedInvGammaPrior


class MixtureofMIGPriors(_MixturePriorBase):
    _prior_class = MultiplicativeInvGammaPrior

