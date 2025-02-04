from scipy.special import betaln
from scipy.special import beta as beta_function
from scipy.stats import beta, binom
from scipy.optimize import minimize
from scipy.stats import gamma, binomtest
from scipy.integrate import quad

import numpy as np
import warnings
import pingouin as pg
from mne.stats.cluster_level import _pval_from_histogram


import numpy as np
import warnings
from scipy.stats import binomtest
from scipy.special import beta as beta_func, betaln
from scipy.optimize import minimize
import pingouin as pg


def bic_to_bf10(bic_h1, bic_h0):
    """
    Convert Bayesian Information Criterion (BIC) values into a Bayes Factor (BF).

    This function computes the Bayes Factor (BF) from the difference in BIC values between two competing models: 
    the alternative hypothesis (H1) and the null hypothesis (H0). The Bayes Factor quantifies the relative evidence 
    for one model over the other. This implementation returns evidence in favor of H0, adjust the order of input accordingly

    The calculation is based on the formula provided by Wagenmakers (2007):

        BF = exp((BIC_H0 - BIC_H1) / 2)

    Parameters:
    -----------
    bic_h1 : float
        The Bayesian Information Criterion (BIC) value for the alternative hypothesis (H1).
    
    bic_h0 : float
        The Bayesian Information Criterion (BIC) value for the null hypothesis (H0).

    Returns:
    --------
    bf01 : float
        The Bayes Factor, representing the relative evidence in favor of H0 compared to H1. A BF > 1 indicates 
        more support for H0, while BF < 1 indicates more support for H0. The magnitude of BF provides a quantitative 
        measure of the strength of evidence.

    References:
    -----------
    Wagenmakers, Eric-Jan. "A practical solution to the pervasive problems of p values." 
    Psychonomic Bulletin & Review 14.5 (2007): 779-804.
    """
    bf01 = np.exp((bic_h1 - bic_h0) / 2)
    return 1/bf01


def beta_binom_ml(k, n, prior):
    """
    Compute the marginal likelihood for a Beta-Binomial model.

    Parameters
    ----------
    k : int
        Number of observed successes.
    n : int
        Number of trials.
    prior : tuple of float
        (alpha, beta) parameters of the Beta prior.

    Returns
    -------
    float
        Marginal likelihood under the Beta-Binomial model.

    Notes
    -----
    The marginal likelihood is given by:
        m = Beta(alpha+k, beta+n-k) / Beta(alpha, beta)
    """
    alpha, beta = prior
    return np.exp(betaln(alpha + k, beta + n - k) - betaln(alpha, beta))


def cap_beta_params(alpha, beta, cap=5000):
    """
    Cap the Beta parameters at a specified maximum, preserving their ratio.

    Parameters
    ----------
    alpha : float
        Alpha parameter of the Beta distribution.
    beta : float
        Beta parameter of the Beta distribution.
    cap : float, optional
        Maximum allowed value for alpha or beta (default=5000).

    Returns
    -------
    alpha_capped, beta_capped : float, float
        The adjusted alpha and beta values, capped but maintaining the ratio alpha:beta.
    """
    max_val = max(alpha, beta)
    if max_val > cap:
        scale = cap / max_val
        alpha_capped = alpha * scale
        beta_capped = beta * scale
        return alpha_capped, beta_capped
    else:
        return alpha, beta


def compute_null_posterior(p, n, alpha_0=1000, beta_0=1000):
    """
    Compute the posterior Beta parameters (alpha_post, beta_post) for an empirical null.

    Interprets each element of p as a probability of success in n trials.
    Summarizes them all by summing across the entire array.

    Parameters
    ----------
    p : ndarray
        Array of empirical null accuracies in [0,1].
    n : int
        Number of trials per shuffle iteration.
    alpha_0 : float, optional
        Prior alpha parameter, by default 1000 (concentrated near p=0.5 if large).
    beta_0 : float, optional
        Prior beta parameter, by default 1000.

    Returns
    -------
    alpha_post : float
        Posterior alpha parameter after incorporating empirical null data.
    beta_post : float
        Posterior beta parameter.

    Notes
    -----
    This approach assumes each entry in `p` is a separate binomial estimate from n trials.
    The posterior is computed by effectively summing the binomial successes and failures
    across all shuffle instances.
    """
    # Calculate the total number of 'throws' across shuffles:
    n_throw = p.shape[0] * n

    # Convert p from frequency to successes:
    alpha_post = np.sum(p * n).astype(int) + alpha_0
    beta_post = beta_0 + n_throw - np.sum(p * n).astype(int)

    return cap_beta_params(alpha_post, beta_post, cap=5000)


def compute_bf(k, n, a, b, p=0.5, alpha_0=1000, beta_0=1000):
    """
    Compute the Bayes factor for a single test location.

    Parameters
    ----------
    k : int
        Number of observed successes.
    n : int
        Number of trials.
    a : float
        Alpha parameter of the alternative prior Beta(a, b).
    b : float
        Beta parameter of the alternative prior Beta(a, b).
    p : float or array-like
        If float, a point null hypothesis at p.
        If array-like, empirical null samples from which a Beta distribution is fitted.

    Returns
    -------
    float
        Bayes factor (BF10).

    Raises
    ------
    ValueError
        If p is array-like and not in [0,1].
    """
    if np.isscalar(p):
        # Point null: use pg.bayesfactor_binom for convenience
        return pg.bayesfactor_binom(k, n, p=p, a=a, b=b)
    else:
        p = np.asarray(p)
        # Fit a Beta distribution to the empirical null
        a_null, b_null = compute_null_posterior(p, n, alpha_0=alpha_0, beta_0=beta_0)
        m0 = beta_binom_ml(k, n, [a_null, b_null])   # null model marginal likelihood
        m1 = beta_binom_ml(k, n, [a, b])             # alternative model marginal likelihood
        return m1 / m0


def pval_accuracy(k, n, p):
    """
    Compute a p-value given observed successes, number of trials, and a null model.

    Parameters
    ----------
    k : int
        Number of observed successes.
    n : int
        Number of trials.
    p : float or array-like
        Null hypothesis definition:
        - If float: a point null; use a binomial test.
        - If array-like: an empirical distribution of accuracies; use _pval_from_histogram.

    Returns
    -------
    float
        P-value under the specified null.
    """
    if np.isscalar(p):
        return binomtest(k, n, p=p).pvalue
    else:
        return _pval_from_histogram([k/n], p, 0)[0]


def bayes_binomtest(k, n, p=0.5, a=1, b=1, verbose=True):
    """
    Compute Bayes factors and p-values from a binomial test using a Beta-Binomial model.

    This function can handle:
    - A single test (k is scalar),
    - Multiple tests in a 1D array,
    - Multiple tests in a 2D array.

    If `p` is a single float, it represents a point null hypothesis at p.
    If `p` is array-like with shape k.shape+(M,), it is interpreted as an empirical null 
    distribution for each test location.

    Parameters
    ----------
    k : float, int, or array-like
        Observed number of successes or observed accuracy.
        If values are between 0 and 1, they will be interpreted as probabilities of success
        and converted to counts by multiplying by n and rounding.
    n : int
        Number of trials.
    p : float or array-like, default=0.5
        Null hypothesis definition:
        - If float: point null at p.
        - If array-like: must have p.shape = k.shape + (M,) for some M. 
          Each slice p[idx] is the null samples for that test location.
    a : float, optional
        Alpha parameter for the alternative Beta prior, by default 1.
    b : float, optional
        Beta parameter for the alternative Beta prior, by default 1.

    Returns
    -------
    BF10 : scalar, ndarray
        Bayes factor(s) for each test location, matching the shape of `k`.
    pvals : scalar, ndarray
        P-value(s) for each test location, matching the shape of `k`.

    Raises
    ------
    ValueError
        If `k` is not 0D, 1D, or 2D.
        If `p` is array-like but does not have p.ndim == k.ndim+1 or p.shape[:k.ndim] != k.shape.
        If k is out of allowed range.

    Notes
    -----
    This function is not intended for combining Bayes factors across subjects. 
    It is designed for single or small sets of values, e.g., decoding accuracies across 
    time or in a time-by-time generalization matrix.
    """
    k = np.asarray(k)

    # Determine how many tests
    if k.ndim == 0:
        n_tests = 1
    else:
        n_tests = np.prod(k.shape)
    if verbose:
        print(f"Conducting {n_tests} binomial Bayes factor test(s).")

    if k.ndim > 2:
        raise ValueError("k must be 0D, 1D, or 2D.")

    # Convert probabilities to counts if needed
    if np.issubdtype(k.dtype, np.floating):
        if np.all((k >= 0) & (k <= 1)):
            successes = k * n
            if not np.allclose(successes, np.round(successes), atol=1e-7):
                warnings.warn("Some values of k*n are not integers. Rounding to nearest integer.", UserWarning)
            successes = np.round(successes).astype(int)
        else:
            raise ValueError("Values must be int or floats between 0 and 1!")
    else:
        successes = k.astype(int)

    shape_of_output = k.shape

    # Handle p array if provided
    if not np.isscalar(p):
        p = np.asarray(p)
        if p.ndim != k.ndim + 1:
            raise ValueError("When p is array-like, p.ndim must be k.ndim+1, representing samples per test.")
        if p.shape[:k.ndim] != k.shape:
            raise ValueError("The shape of p (except last dim) must match k.shape.")
        if verbose:
            print('Using null distribution to estimate H0 prior')

    # Special case: if k is 0D (a single value)
    if k.ndim == 0:
        # Scalar case
        bf = compute_bf(int(successes), n, a, b, p=p if not np.isscalar(p) else p)
        pval = pval_accuracy(int(successes), n, p if not np.isscalar(p) else p)
        return bf, pval

    # For k.ndim == 1 or k.ndim == 2, handle with a single code path.
    BF_out = np.zeros(shape_of_output)
    pval_out = np.zeros(shape_of_output)

    # Iterate over all test locations using np.ndindex
    for idx in np.ndindex(shape_of_output):
        k_succ = int(successes[idx])
        if np.isscalar(p):
            # Point null
            local_p = p
        else:
            # Empirical null for this location is p[idx], a 1D array of null samples
            local_p = p[idx]

        BF_out[idx] = compute_bf(k_succ, n, a, b, p=local_p)
        pval_out[idx] = pval_accuracy(k_succ, n, local_p)

    return BF_out, pval_out


def bayes_ttest(x, y=0, paired=False, alternative='two-sided', r=0.707, return_pval=False):
    """
    Compute Bayes Factors from a t-test using Pingouin's JZS method, applied to 
    data arrays that can be 1D, 2D, or 3D. 
    
    Parameters
    ----------
    x : array
        Can be of shape:
         - (n): n observations
         - (n, t): n observations at each time point
         - (n, t, freq): n observations at each time-frequency "pixel"
         - (n, t, t): n observations at each time-by-time "pixel"
    y : array or scalar, default=0
        If scalar, a one-sample t-test (x vs mu) is performed.
        If array and same shape as x (except first dim), a two-sample test is performed.
    paired : bool, default=False
        If True, perform a paired t-test (requires x and y to have identical shape).
    alternative : str, default='two-sided'
        Defines the alternative hypothesis. Must be 'two-sided', 'greater', or 'less'.
    r : float, default=0.707
        Prior width for the JZS Bayes factor computation.

    Returns
    -------
    results : dict
        A dictionary containing:
        - 'T': array of t-values
        - 'pval': array of p-values
        - 'BF10': array of Bayes factors (BF10)
        
        If x is 1D, these are scalars.
        If x is multi-dimensional (2D, 3D), these are arrays of shape 
        matching the non-observation dimensions of x.
    """
    x = np.asarray(x)
    
    # Check if y is a scalar or array
    if np.isscalar(y):
        # One-sample test scenario
        y_is_scalar = True
        y_val = float(y)
    else:
        y_is_scalar = False
        y = np.asarray(y)

    # Basic input checks
    if x.ndim < 1 or x.ndim > 3:
        raise ValueError(f"x must be 1D, 2D, or 3D, but got {x.ndim}D")

    # If two-sample, ensure shape compatibility
    if not y_is_scalar:
        if x.shape != y.shape:
            raise ValueError("For a two-sample test, x and y must have the same shape.")
    
    # Check paired requirement
    if paired and y_is_scalar:
        raise ValueError("For a paired test, y must be an array, not a scalar.")
    
    # Determine test size and print info
    # We always have n in dimension 0
    shape_without_n = x.shape[1:]
    if y_is_scalar:
        test_type = "one-sample"
    else:
        if paired:
            test_type = "paired"
        else:
            test_type = "two-sample"
    
    # Determine how many tests we run
    if x.ndim == 1:
        # single test
        n_tests = 1
        shape_of_output = ()
    else:
        # multiple tests
        shape_of_output = shape_without_n
        n_tests = np.prod(shape_without_n)
    
    print(f"We will conduct a {test_type} t-test for {n_tests} point(s).")

    # Prepare result containers
    if n_tests == 1:
        res = pg.ttest(x, y, paired=paired, 
                alternative=alternative, r=r)
        if return_pval:
            return res['BF10'].values[0], res['p-val'].values[0]
        else:
            return res['BF10'].values[0]
    else:
        # Multiple tests: loop over the shape and run test for each slice
        BF_out = np.zeros(shape_of_output)
        if return_pval:
            pval_out = np.zeros(shape_of_output)
        # We will iterate over all indices in shape_without_n using np.ndindex
        for idx in np.ndindex(shape_of_output):
            # Construct slice
            # For (n), idx = () empty
            # For (n, t), idx = (time_point,)
            # For (n, t, freq), idx = (time_point, freq_point)
            # The data slice will be x[:, idx...]
            # where idx can be used directly inside indexing
            x_slice = x[(slice(None),) + idx]
            if not y_is_scalar:
                y_slice = y[(slice(None),) + idx]
            else:
                y_slice = y

            # run the test
            res = pg.ttest(x_slice, y_slice, paired=paired, alternative=alternative, r=r)
            BF_out[idx] = res['BF10'].values[0]
            if return_pval:
                pval_out[idx] = res['p-val'].values[0]
        if return_pval:
            return BF_out, pval_out
        else:
            return BF_out


def sim_decoding_binomial(t0, tmax, sfreq, scale_factor=3, tstart=0, ntrials=100):
    """
    Simulate a 1D time series of decoding accuracy using a binomial distribution.

    Parameters
    ----------
    t0 : int
        Start time of the simulation (in seconds).
    tmax : int
        End time of the simulation (in seconds).
    sfreq : int
        Sampling frequency (number of samples per second).
    scale_factor : float, optional
        Factor to normalize the decoding accuracy. Default is 3.
    tstart : float, optional
        Location parameter for the gamma distribution. Default is 0.
    ntrials : int, optional
        Number of trials for the binomial distribution. Default is 100.

    Returns
    -------
    numpy.ndarray
        Simulated number of successes (k) for each time point in the time series.
    """
    # Simulate a time series of decoding accuracy:
    times = np.linspace(t0, tmax, sfreq * (tmax - t0))
    # Create a time series of decoding accuracy:
    decoding_accuracy = gamma.pdf(times, 3, scale=0.2, loc=tstart)
    # Normalize it:
    decoding_accuracy_true = 0.5 + (decoding_accuracy/np.max(decoding_accuracy))/scale_factor

    # Loop through each time points:
    k = np.random.binomial(ntrials, decoding_accuracy_true)  # number of successes

    return k



def sim_decoding_binomial_2d(t0, tmax, sfreq, scale_factor=3, tstart=0, ntrials=100):
    """
    Simulate a 2D matrix of decoding accuracy using a binomial distribution.

    Parameters
    ----------
    t0 : int
        Start time of the simulation (in seconds).
    tmax : int
        End time of the simulation (in seconds).
    sfreq : int
        Sampling frequency (number of samples per second).
    scale_factor : float, optional
        Factor to normalize the decoding accuracy matrix. Default is 3.
    tstart : float, optional
        Location parameter for the gamma distribution. Default is 0.
    ntrials : int, optional
        Number of trials for the binomial distribution. Default is 100.

    Returns
    -------
    numpy.ndarray
        Simulated 2D array of binomial observations for decoding accuracy.
    """
    # Simulate a time series of decoding accuracy:
    times = np.linspace(t0, tmax, sfreq * (tmax - t0))
    # Create a time series of decoding accuracy:
    decoding_accuracy = gamma.pdf(times, 3, scale=0.2, loc=tstart)
    # Calcuate the probability matrix:
    prob_matrix = np.outer(decoding_accuracy, decoding_accuracy)

    # Normalize it:
    prob_matrix_norm = 0.5 + (prob_matrix/np.max(prob_matrix))/scale_factor

    # Create observation
    obs = np.random.binomial(n=ntrials, p=prob_matrix_norm)
    return obs

from scipy.special import beta as beta_func
from scipy.stats import norm


def scaled_beta_tau(tau, alpha=1.0, beta=1.0):
    """
    Compute the scaled Beta prior for Kendall's tau.

    Parameters
    ----------
    tau : float or ndarray
        The Kendall's tau value(s) at which to evaluate the prior.
    alpha : float, optional
        The alpha parameter for the Beta function, by default 1.0.
    beta : float, optional
        The beta parameter for the Beta function, by default 1.0.

    Returns
    -------
    float or ndarray
        The value(s) of the scaled beta prior at tau.
    """
    tau = np.atleast_1d(tau)
    val = ((np.pi * 2 ** (-2 * alpha)) / beta_func(alpha, alpha)) * (
        np.cos((np.pi * tau) / 2) ** (2 * alpha - 1)
    )
    return val if val.size > 1 else val[0]


def prior_tau(tau, kappa=1.0):
    """
    Compute the prior distribution p(tau) for two-sided tests.

    p(tau) = scaledBetaTau(tau, alpha=1/kappa, beta=1/kappa)

    Parameters
    ----------
    tau : float or ndarray
        Kendall's tau value(s).
    kappa : float, optional
        The parameter that controls the prior shape, by default 1.0.

    Returns
    -------
    float or ndarray
        Prior density at tau.
    """
    return scaled_beta_tau(tau, alpha=1.0 / kappa, beta=1.0 / kappa)


def prior_tau_positive(tau, kappa=1.0):
    """
    Compute the one-sided prior distribution p(tau) for tau in [0,1].

    Parameters
    ----------
    tau : float or ndarray
        Kendall's tau value(s).
    kappa : float, optional
        The parameter controlling the shape of the prior, by default 1.0.

    Returns
    -------
    float or ndarray
        Prior density at tau (0 if tau outside [0,1]).
    """
    tau = np.atleast_1d(tau)
    result = np.zeros_like(tau)
    idx = (tau >= 0) & (tau <= 1)
    result[idx] = 2 * prior_tau(tau[idx], kappa)
    return result if result.size > 1 else result[0]


def prior_tau_negative(tau, kappa=1.0):
    """
    Compute the one-sided prior distribution p(tau) for tau in [-1,0].

    Parameters
    ----------
    tau : float or ndarray
        Kendall's tau value(s).
    kappa : float, optional
        The parameter controlling the shape of the prior, by default 1.0.

    Returns
    -------
    float or ndarray
        Prior density at tau (0 if tau outside [-1,0]).
    """
    tau = np.atleast_1d(tau)
    result = np.zeros_like(tau)
    idx = (tau >= -1) & (tau <= 0)
    result[idx] = 2 * prior_tau(tau[idx], kappa)
    return result if result.size > 1 else result[0]


def post_density_kendall_tau(tau, T_star, n, kappa=1.0, var=1.0, test="two-sided"):
    """
    Compute the unnormalized posterior density for Kendall's tau.

    The posterior is proportional to p(T_star|tau)*p(tau).

    Parameters
    ----------
    tau : float or ndarray
        Kendall's tau value(s).
    T_star : float
        Standardized Kendall's tau statistic.
    n : int
        Sample size.
    kappa : float, optional
        Parameter controlling the prior shape, by default 1.0.
    var : float, optional
        Variance parameter for the likelihood. min(var,1) is used, by default 1.0.
    test : str, optional
        Type of test: "two-sided", "positive", or "negative". By default "two-sided".

    Returns
    -------
    float or ndarray
        Unnormalized posterior density at tau.
    """
    tau = np.atleast_1d(tau)
    var = min(1.0, var)

    # Select prior function based on test type
    if test == "two-sided":
        prior_func = prior_tau
    elif test == "positive":
        prior_func = prior_tau_positive
    elif test == "negative":
        prior_func = prior_tau_negative
    else:
        raise ValueError("test must be 'two-sided', 'positive', or 'negative'.")

    # Likelihood for T* given tau
    likelihood = norm.pdf(T_star, loc=1.5 * tau * np.sqrt(n), scale=np.sqrt(var))
    posterior_unnormalized = likelihood * prior_func(tau)

    return posterior_unnormalized if posterior_unnormalized.size > 1 else posterior_unnormalized[0]


def posterior_tau(tau, kentau, n, kappa=1.0, var=1.0, test="two-sided"):
    """
    Compute the normalized posterior density p(tau|data).

    Parameters
    ----------
    tau : float or ndarray
        Kendall's tau value(s) at which to evaluate the posterior.
    kentau : float
        Observed Kendall's tau.
    n : int
        Sample size.
    kappa : float, optional
        Parameter controlling prior shape, by default 1.0.
    var : float, optional
        Variance parameter for the likelihood (min(1,var) used), by default 1.0.
    test : str, optional
        Type of test: "two-sided", "positive", or "negative", by default "two-sided".

    Returns
    -------
    float or ndarray
        The posterior density at tau.
    """
    tau = np.atleast_1d(tau)
    var = min(1.0, var)

    # Compute T*
    T_star = (kentau * ((n * (n - 1)) / 2.0)) / np.sqrt(n * (n - 1) * (2 * n + 5) / 18.0)

    # Integration limits based on test
    if test == "two-sided":
        lims = (-1, 1)
    elif test == "positive":
        lims = (0, 1)
    elif test == "negative":
        lims = (-1, 0)
    else:
        raise ValueError("test must be 'two-sided', 'positive', or 'negative'.")

    def integrand(x):
        return post_density_kendall_tau(x, T_star, n, kappa, var, test=test)

    integral_val, _ = quad(integrand, lims[0], lims[1])

    post_vals = post_density_kendall_tau(tau, T_star, n, kappa, var, test=test) / integral_val
    return post_vals


def bf_kendall_tau(tau, n, kappa=1.0, var=1.0):
    """
    Compute Bayes factors for Kendall's tau.

    Parameters
    ----------
    tau : float
        Observed Kendall's tau.
    n : int
        Sample size.
    kappa : float, optional
        Parameter controlling the prior shape, by default 1.0.
    var : float, optional
        Variance parameter for the likelihood, by default 1.0.

    Returns
    -------
    dict
        A dictionary with keys 'n', 'r', 'bf10', 'bfPlus0', and 'bfMin0' representing sample size,
        observed tau, and the three Bayes factors respectively.
    """
    result = {
        'n': n,
        'r': tau,
        'bf10': None,
        'bfPlus0': None,
        'bfMin0': None
    }

    # BF10 (two-sided)
    result['bf10'] = prior_tau(0, kappa) / posterior_tau(0, tau, n, kappa=kappa, var=var, test="two-sided")

    # BF+0 (one-sided, positive)
    result['bfPlus0'] = prior_tau_positive(0, kappa) / posterior_tau(0, tau, n, kappa=kappa, var=var, test="positive")

    # BF-0 (one-sided, negative)
    result['bfMin0'] = prior_tau_negative(0, kappa) / posterior_tau(0, tau, n, kappa=kappa, var=var, test="negative")

    return result


def kendall_bf_from_data(x, y, kappa=1.0, var=1.0):
    """
    Compute Bayes factors for Kendall's tau given raw data.

    Parameters
    ----------
    x : array_like
        Data vector for variable X.
    y : array_like
        Data vector for variable Y.
    kappa : float, optional
        Parameter controlling the prior shape, by default 1.0.
    var : float, optional
        Variance parameter for the likelihood, by default 1.0.

    Returns
    -------
    tuple
        (tau, bf_result) where tau is the Kendall's tau and bf_result is a dictionary of Bayes factors.
    """
    res = pg.corr(x, y, method='kendall')
    tau_val = res['r'].values[0]
    n = len(x)
    bf_result = bf_kendall_tau(tau_val, n, kappa=kappa, var=var)
    return tau_val, res['p-val'].values[0], bf_result


if __name__ == "__main__":
    # Example usage
    yourKendallTauValue = -0.3
    yourN = 20

    # Compute BFs with given tau and N
    bf_result = bf_kendall_tau(yourKendallTauValue, yourN)
    print("Bayes Factors:", bf_result)
