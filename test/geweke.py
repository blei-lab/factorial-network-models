import numpy as np
import numpy.random as npr

from tqdm import tqdm

import importlib
import lsm.models
importlib.reload(lsm.models)
from lsm.models import LatentSpaceModel, MixtureOfLatentSpaceModels, FactorialLatentSpaceModel

import lsm.priors
importlib.reload(lsm.priors)
from lsm.priors import SharedInvGammaPrior, MultiplicativeInvGammaPrior


def geweke_test_ig(K, N=10, a=2.5, b=1.5, N_iter=100000, tol=1e-1):
    prior = SharedInvGammaPrior(K, a=a, b=b)
    X = np.sqrt(prior.sigmasq) * npr.randn(N, K)

    nus = [prior.nu]
    Xs = [X.copy()]
    for _ in tqdm(range(N_iter)):
        prior.resample(X)
        X = np.sqrt(prior.sigmasq) * npr.randn(N, K)

        nus.append(prior.nu)
        Xs.append(X.copy())

    # Compute prior stats
    pri_E_nu = a / b
    pri_std_nu = np.sqrt(a / b**2)

    # Compute empirical stats
    emp_E_nu = np.mean(nus, axis=0)
    emp_std_nu = np.std(nus, axis=0)

    print("prior E[nu]:       ", pri_E_nu)
    print("empirical E[nu]:   ", emp_E_nu)
    print("")
    print("prior std[nu]:     ", pri_std_nu)
    print("empirical std[nu]: ", emp_std_nu)

    assert np.all(abs(pri_E_nu - emp_E_nu) < tol)


def geweke_test_mig(K, V=10, a1=2.5, a2=3.5, N_iter=10000, tol=1e-1):
    prior = MultiplicativeInvGammaPrior(K, a1=a1, a2=a2)
    X = np.sqrt(prior.sigmasq) * npr.randn(V, K)

    log_joint = lambda: \
        -0.5 * np.sum(np.log(2 * np.pi * prior.sigmasq)) * V + \
        -0.5 * np.sum(X**2 / prior.sigmasq) + \
        prior.log_prior()

    nus = [prior.nu.copy()]
    Xs = [X.copy()]
    lps = [log_joint()]
    for _ in tqdm(range(N_iter)):
        prior.resample(X)
        X = np.sqrt(prior.sigmasq) * npr.randn(V, K)

        nus.append(prior.nu.copy())
        Xs.append(X.copy())
        lps.append(log_joint())

    import matplotlib.pyplot as plt
    plt.plot(lps[::max(N_iter//1000, 1)])
    plt.show()

    # Compute prior stats
    pri_E_nu = np.hstack(([a1], a2 * np.ones(K-1)))
    pri_std_nu = np.sqrt(pri_E_nu)

    # Compute empirical stats
    emp_E_nu = np.mean(nus, axis=0)
    emp_std_nu = np.std(nus, axis=0)

    print("prior E[nu]:       ", pri_E_nu.round(2))
    print("empirical E[nu]:   ", emp_E_nu.round(2))
    print("")
    print("prior std[nu]:     ", pri_std_nu.round(2))
    print("empirical std[nu]: ", emp_std_nu.round(2))

    assert np.all(abs(pri_E_nu - emp_E_nu) < tol)



def geweke_test_lsm(K, V=5, N=2, N_iter=10000, tol=1e-1):
    lsm = LatentSpaceModel(V, K, sigmasq_b=1.0)
    lsm.sigmasq_x_prior.nu = 1.0

    Xs = [lsm.X.copy()]
    bs = [lsm.b.copy()]
    As = [np.array([lsm.generate() for _ in range(N)])]
    for _ in tqdm(range(N_iter)):
        # Resample model X and b
        lsm._resample_X()
        lsm._resample_b()
        Xs.append(lsm.X.copy())
        bs.append(lsm.b.copy())

        # Remove old data
        lsm.As = []
        lsm.masks = []
        lsm.ms = []

        # Generate new data
        As.append(np.array([lsm.generate() for _ in range(N)]))

    # Compute empirical stats
    emp_E_X = np.mean(Xs)
    emp_std_X = np.std(Xs)

    emp_E_b = np.mean(bs)
    emp_std_b = np.std(bs)

    print("prior E[X]:       ", 0)
    print("empirical E[X]:   ", emp_E_X.round(3))
    print("")
    print("prior std[X]:     ", 1)
    print("empirical std[X]: ", emp_std_X.round(3))
    print("")
    print("prior E[b]:       ", 0)
    print("empirical E[b]:   ", emp_E_b.round(3))
    print("")
    print("prior std[b]:     ", 1)
    print("empirical std[b]: ", emp_std_b.round(3))

    assert np.all(abs(emp_E_X) < tol)
    assert np.all(abs(emp_E_b) < tol)


def geweke_test_molsm(K, H, V=5, N=2, N_iter=10000, tol=1e-1):
    """
    Just test the mixture part.  The X and b sampling is same as LSM.
    """
    molsm = MixtureOfLatentSpaceModels(V, K, H, sigmasq_b=1.0)
    molsm.sigmasq_x_prior.nu = 1.0

    hs = [np.bincount(molsm.hs, minlength=H)]
    nus = [molsm.nu]
    As = [np.array([molsm.generate() for _ in range(N)])]
    for _ in tqdm(range(N_iter)):
        # Resample model X and b
        molsm.resample()
        hs.append(np.bincount(molsm.hs, minlength=H))
        nus.append(molsm.nu.copy())

        # Remove old data
        molsm.As = []
        molsm.masks = []
        molsm.ms = []
        molsm.mhs = []

        # Generate new data
        As.append(np.array([molsm.generate() for _ in range(N)]))

    # Determine prior
    prior_E_h = N / float(H) * np.ones(H)
    prior_E_nu = 1 / float(H) * np.ones(H)

    # Compute empirical stats
    emp_E_h = np.mean(hs, axis=0)
    emp_E_nu = np.mean(nus, axis=0)

    print("prior E[h]:       ", prior_E_h)
    print("empirical E[h]:   ", emp_E_h.round(3))
    print("")
    print("prior E[nu]:      ", prior_E_nu)
    print("empirical E[nu]:  ", emp_E_nu.round(3))

    assert np.all(abs(emp_E_h - prior_E_h) < tol)
    assert np.all(abs(emp_E_nu - prior_E_nu) < tol)


def geweke_test_flsm(K, V=5, N=5, alpha=1.0, N_iter=10000, tol=2e-1):
    """
    Just test the mixture part.  The X and b sampling is same as LSM.
    """
    flsm = FactorialLatentSpaceModel(V, K, sigmasq_b=1.0, alpha=alpha)
    flsm.sigmasq_x_prior.nu = 1.0

    As = [np.array([flsm.generate() for _ in range(N)])]
    rhos = [flsm.rho.copy()]
    msums = [np.sum(flsm.ms, axis=0)]
    for _ in tqdm(range(N_iter)):
        # Resample model
        flsm.resample()
        msums.append(np.sum(flsm.ms, axis=0))
        rhos.append(flsm.rho.copy())

        # Remove old data
        flsm.As = []
        flsm.masks = []
        flsm.ms = []

        # Generate new data
        As.append(np.array([flsm.generate() for _ in range(N)]))

    # Determine prior
    prior_E_rho = (alpha / K) / (alpha / K + 1.0)
    prior_E_msum = N * prior_E_rho

    # Compute empirical stats
    emp_E_rho = np.mean(rhos, axis=0)
    emp_E_msum = np.mean(msums, axis=0)

    print("prior E[rho]:       ", prior_E_rho)
    print("empirical E[rho]:   ", emp_E_rho.round(3))
    print("")
    print("prior E[msum]:      ", prior_E_msum)
    print("empirical E[msum]:  ", emp_E_msum.round(3))

    assert np.all(abs(emp_E_rho - prior_E_rho) < tol)
    assert np.all(abs(emp_E_msum - prior_E_msum) < tol)

if __name__ == "__main__":
    # geweke_test_ig(5)
    geweke_test_mig(5)
    # geweke_test_lsm(2)
    # geweke_test_molsm(10, 5)
    # geweke_test_flsm(5)
