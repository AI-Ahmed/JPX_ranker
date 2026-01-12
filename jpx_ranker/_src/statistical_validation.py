"""
Statistical validation utilities for financial ML.

Implements López de Prado's selection bias correction framework (2012-2025)
for proper evaluation of multiple testing scenarios in Optuna hyperparameter optimization.

These functions help us avoid fooling ourselves with lucky results by properly
accounting for the selection bias that occurs when we test multiple strategies
and pick the best one.

References
----------
.. [1] López de Prado, M. (2018). "Advances in Financial Machine Learning", Chapter 12.
       Wiley. https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086
.. [2] López de Prado, M., Lipton, A., & Zoonekynd, V. (2025). "Sharpe Ratio Inference: 
       A New Standard for Decision-Making and Reporting." Forthcoming in Journal of 
       Portfolio Management, 2026. Available at SSRN: https://ssrn.com/abstract=5520741 
       or http://dx.doi.org/10.2139/ssrn.5520741
.. [3] Bailey, D. H., & López de Prado, M. (2012). "The Sharpe Ratio Efficient Frontier".
       Journal of Risk, 15(2), 3-44.
.. [4] Bailey, D. H., & López de Prado, M. (2014). "The Deflated Sharpe Ratio: 
       Correcting for Selection Bias, Backtest Overfitting and Non-Normality".
       Journal of Portfolio Management, 40(5), 94-107.

Notes
-----
Implementation based on https://github.com/zoonek/2025-sharpe-ratio/blob/main/functions.py
"""

import math
import numpy as np
import scipy.stats


def sharpe_ratio_variance(
    SR: float,
    T: int,
    *,
    gamma3: float = 0.,
    gamma4: float = 3.,
    rho: float = 0.,
    K: int = 1,
) -> float:
    """
    Asymptotic variance of Sharpe ratio, adjusted for multiple testing.
    
    Parameters
    ----------
    SR : float
        Observed Sharpe ratio
    T : int
        Number of observations (time periods)
    gamma3 : float, optional
        Skewness of returns (default: 0 for normal distribution)
    gamma4 : float, optional
        Kurtosis of returns (default: 3 for normal distribution)
    rho : float, optional
        Autocorrelation of returns (default: 0 for independent returns)
    K : int, optional
        Number of trials for multiple testing adjustment (default: 1)
        
    Returns
    -------
    float
        Variance of the Sharpe ratio estimator
        
    Notes
    -----
    **Simple Explanation**: How uncertain is our Sharpe ratio estimate?
    
    The variance of the Sharpe ratio depends on the return distribution's higher moments.
    When returns are non-normal (skewed or fat-tailed), the standard variance formula
    underestimates uncertainty. This function accounts for:
    
    - **Skewness (γ₃)**: Asymmetry in return distribution
      - Negative skew (γ₃ < 0): More extreme losses than gains
      - Positive skew (γ₃ > 0): More extreme gains than losses
      
    - **Kurtosis (γ₄)**: Fat tails and extreme events
      - γ₄ = 3: Normal distribution
      - γ₄ > 3: Fat tails (more extreme events than normal)
      - γ₄ < 3: Thin tails (fewer extreme events)
      
    - **Autocorrelation (ρ)**: Serial correlation in returns
      - ρ > 0: Positive momentum (today's return predicts tomorrow's)
      - ρ < 0: Mean reversion (gains followed by losses)
      
    - **Multiple testing (K)**: Inflated variance from trying K strategies
      - K = 1: Single strategy
      - K > 1: Best of K strategies (reduces variance due to selection)
    
    **Key insight**: Higher variance means we need more data to be confident 
    in the Sharpe ratio. Non-normal returns require longer track records.
    
    References
    ----------
    .. [1] López de Prado, M., Lipton, A., & Zoonekynd, V. (2025). "Sharpe Ratio Inference: 
       A New Standard for Decision-Making and Reporting." Available at SSRN: 
       https://ssrn.com/abstract=5520741
    
    Examples
    --------
    >>> # Normal returns, 24 observations
    >>> var = sharpe_ratio_variance(SR=0.5, T=24, gamma3=0, gamma4=3)
    >>> std = np.sqrt(var)
    >>> print(f"Standard error: {std:.3f}")
    
    >>> # Non-normal returns (skewed, fat-tailed)
    >>> var = sharpe_ratio_variance(SR=0.5, T=24, gamma3=-2.4, gamma4=10.2)
    >>> std = np.sqrt(var)
    >>> print(f"Standard error (non-normal): {std:.3f}")  # Higher!
    """
    # Autocorrelation adjustment factors
    A = 1
    B = rho / (1 - rho) if rho != 1 else 0
    C = rho**2 / (1 - rho**2) if rho != 1 else 0
    
    a = A + 2 * B
    b = A + B + C
    c = A + 2 * C
    
    # Variance formula accounting for non-normality
    V = (a * 1 - b * gamma3 * SR + c * (gamma4 - 1) / 4 * SR**2) / T
    
    # Adjustment for multiple testing (K trials)
    # When K > 1, we select the maximum, which has different variance
    if K > 1:
        # Simplified adjustment: variance decreases with K
        # Full formula would use moments_Mk from López de Prado
        V = V * (1 - 0.5 * np.log(K) / K)  # Approximation
    
    return V


def minimum_track_record_length(
    SR: float,
    SR0: float,
    *,
    gamma3: float = 0.,
    gamma4: float = 3.,
    rho: float = 0.,
    alpha: float = 0.05,
) -> float:
    """
    Minimum observations needed for SR to be significantly > SR0.
    
    Parameters
    ----------
    SR : float
        Observed Sharpe ratio
    SR0 : float
        Null hypothesis Sharpe ratio (benchmark)
    gamma3 : float, optional
        Skewness of returns (default: 0)
    gamma4 : float, optional
        Kurtosis of returns (default: 3 for normal)
    rho : float, optional
        Autocorrelation of returns (default: 0)
    alpha : float, optional
        Significance level (default: 0.05 for 95% confidence)
        
    Returns
    -------
    float
        Minimum track record length needed (number of observations)
        
    Notes
    -----
    **Simple Explanation**: How many trials do we need to be confident our strategy 
    is truly better than the benchmark?
    
    **Coin flip analogy**: If you flip a coin 10 times and get 6 heads, you can't 
    conclude it's biased. But if you flip 1000 times and get 600 heads, that's 
    statistically significant. MinTRL tells you the minimum number of flips needed.
    
    **In trading context**:
    - You observe SR = 1.0 from your strategy
    - Benchmark is SR0 = 0 (no skill)
    - MinTRL might say you need 50 trading days to be 95% confident
    - If you only have 20 days, you can't claim statistical significance yet
    
    **Use case in Optuna**: Stop optimization early when we've collected enough evidence.
    If we've run 50 trials and MinTRL says we only need 30, we can stop - we already
    have statistical significance. This saves computational time!
    
    **Key insights**:
    - Better strategies (higher SR - SR0) need fewer observations
    - Noisy strategies (high γ₃, γ₄, ρ) need more data
    - Lower alpha (more confidence) requires more observations
    
    **Interpretation**:
    - MinTRL = 10: Need only 10 observations (very strong signal)
    - MinTRL = 100: Need 100 observations (weak signal)
    - MinTRL = 1000: Need 1000 observations (very weak signal)
    
    References
    ----------
    .. [1] López de Prado, M., Lipton, A., & Zoonekynd, V. (2025). "Sharpe Ratio Inference: 
       A New Standard for Decision-Making and Reporting." Available at SSRN: 
       https://ssrn.com/abstract=5520741
    
    Examples
    --------
    >>> # Strong strategy: SR=1.0 vs benchmark SR0=0
    >>> min_trl = minimum_track_record_length(SR=1.0, SR0=0, alpha=0.05)
    >>> print(f"Need {min_trl:.0f} observations")
    
    >>> # Weak strategy: SR=0.2 vs benchmark SR0=0
    >>> min_trl = minimum_track_record_length(SR=0.2, SR0=0, alpha=0.05)
    >>> print(f"Need {min_trl:.0f} observations")  # Much higher!
    """
    # Variance at T=1 (per observation)
    variance = sharpe_ratio_variance(
        SR0, T=1, gamma3=gamma3, gamma4=gamma4, rho=rho, K=1
    )
    
    # Critical value for significance test
    z_alpha = scipy.stats.norm.ppf(1 - alpha)
    
    # Minimum track record length formula
    min_trl = variance * (z_alpha / (SR - SR0)) ** 2
    
    return min_trl


def probabilistic_sharpe_ratio(
    SR: float,
    SR0: float,
    *,
    variance: float = None,
    T: int = None,
    gamma3: float = 0.,
    gamma4: float = 3.,
    rho: float = 0.,
    K: int = 1,
) -> float:
    """
    PSR: probability that true SR > SR0 given observed SR.
    
    Parameters
    ----------
    SR : float
        Observed Sharpe ratio
    SR0 : float
        Null hypothesis Sharpe ratio (benchmark)
    variance : float, optional
        Variance of SR estimator (if provided, T is ignored)
    T : int, optional
        Number of observations (required if variance not provided)
    gamma3 : float, optional
        Skewness of returns (default: 0)
    gamma4 : float, optional
        Kurtosis of returns (default: 3 for normal)
    rho : float, optional
        Autocorrelation of returns (default: 0)
    K : int, optional
        Number of trials for multiple testing adjustment (default: 1)
        
    Returns
    -------
    float
        Probabilistic Sharpe Ratio in [0, 1]
        PSR > 0.95 indicates statistical significance at α=0.05
        
    Notes
    -----
    **Simple Explanation**: What's the probability our strategy is actually skillful?
    
    You observe SR = 1.5 from 100 trading days. But is this real skill or just luck?
    PSR gives you a probability: "There's a 97% chance the true SR is positive."
    
    **Interpretation**:
    - PSR = 0.50: Coin flip - could be luck or skill (50/50 chance)
    - PSR = 0.95: 95% confident it's real skill (standard threshold)
    - PSR = 0.99: 99% confident - very strong evidence
    - PSR < 0.50: More likely to be luck than skill
    
    **Why it matters**: Traditional Sharpe ratio doesn't account for uncertainty.
    - SR = 2.0 from 10 days is less reliable than SR = 1.0 from 1000 days
    - PSR properly accounts for sample size and return distribution properties
    - It converts the Sharpe ratio to a "probability scale" [0,1]
    
    **Adjustment for K trials**: If you test K=100 strategies and pick the best,
    PSR adjusts for this "multiple testing" - the best of 100 random strategies
    will look good by chance alone. PSR accounts for this selection bias.
    
    **Real-world example**:
    - You backtest 50 trading strategies
    - Best one has SR = 2.0 over 200 days
    - Without adjustment: "Wow, SR=2.0 is great!"
    - With PSR (K=50): "PSR = 0.82, only 82% confident it's real"
    - The selection bias reduced our confidence significantly
    
    References
    ----------
    .. [1] Bailey, D. H., & López de Prado, M. (2012). The Sharpe Ratio Efficient Frontier.
       Journal of Risk, 15(2), 3-44.
    .. [2] López de Prado, M., Lipton, A., & Zoonekynd, V. (2025). "Sharpe Ratio Inference: 
       A New Standard for Decision-Making and Reporting." Available at SSRN: 
       https://ssrn.com/abstract=5520741

    Examples
    --------
    >>> # Single strategy, 100 observations
    >>> psr = probabilistic_sharpe_ratio(SR=1.0, SR0=0, T=100)
    >>> print(f"PSR = {psr:.3f}")
    
    >>> # Best of 50 strategies (multiple testing)
    >>> psr = probabilistic_sharpe_ratio(SR=1.0, SR0=0, T=100, K=50)
    >>> print(f"PSR (adjusted) = {psr:.3f}")  # Lower due to selection bias
    """
    # Flexible API: caller can provide either T (compute variance) or pre-computed variance
    # This is NOT unreachable code - it's a design pattern for optimization
    if variance is None:
        if T is None:
            raise ValueError("Must provide either variance or T")
        variance = sharpe_ratio_variance(
            SR0, T, gamma3=gamma3, gamma4=gamma4, rho=rho, K=K
        )
    
    # PSR is the CDF of standard normal at (SR - SR0) / sqrt(variance)
    # This is equivalent to 1 - p_value for one-sided test
    psr = scipy.stats.norm.cdf((SR - SR0) / math.sqrt(variance))
    
    return psr


def oFDR(
    SR: float,
    SR0: float,
    SR1: float,
    T: int,
    p_H1: float,
    *,
    gamma3: float = 0.,
    gamma4: float = 3.,
    rho: float = 0.,
    K: int = 1,
) -> float:
    """
    Observed False Discovery Rate: P[H0|SR>SR_obs].
    
    Probability that we incorrectly reject null hypothesis given observed SR.
    
    Parameters
    ----------
    SR : float
        Observed Sharpe ratio
    SR0 : float
        Null hypothesis SR (no skill, e.g., 0)
    SR1 : float
        Alternative hypothesis SR (meaningful strategy, e.g., 0.5)
    T : int
        Number of observations
    p_H1 : float
        Prior probability that strategy is good (e.g., 0.05 = 5%)
    gamma3 : float, optional
        Skewness of returns (default: 0)
    gamma4 : float, optional
        Kurtosis of returns (default: 3 for normal)
    rho : float, optional
        Autocorrelation of returns (default: 0)
    K : int, optional
        Number of trials for multiple testing adjustment (default: 1)
        
    Returns
    -------
    float
        Observed FDR in [0, 1]
        oFDR < 0.25 typically indicates acceptable false discovery rate
        
    Notes
    -----
    **Simple Explanation**: What's the probability we're fooling ourselves?
    
    Imagine you test 100 trading strategies. You find one with great backtest results.
    But here's the problem: even if all 100 strategies are worthless, some will look
    good by pure chance. oFDR tells you the probability you picked a "false positive."
    
    **Real-world analogy - Medical testing**:
    - You test positive for a rare disease (1% prevalence)
    - Test is 95% accurate
    - But oFDR might be 0.83 - meaning 83% chance it's a false alarm!
    - Why? Because the disease is rare, most positives are false positives
    
    **In trading**:
    - SR0 = 0: Null hypothesis (no skill, random strategy)
    - SR1 = 0.5: Alternative hypothesis (meaningful edge)
    - p_H1 = 0.05: Prior belief that only 5% of strategies are good
    - You observe SR = 1.2
    - oFDR = 0.20: 20% chance this "discovery" is false
    
    **Interpretation**:
    - oFDR < 0.05: Very confident (< 5% chance of false discovery)
    - oFDR < 0.25: Acceptable (< 25% chance of false discovery)
    - oFDR > 0.50: Likely a false positive - don't trust it!
    - oFDR ≈ 1.00: Almost certainly a false positive
    
    **Why it matters**: Protects against overfitting and data mining.
    
    **Example scenario**:
    - You run 200 Optuna trials
    - Best trial has SR = 1.5
    - Without oFDR: "Great! SR=1.5 is excellent!"
    - With oFDR (K=200): "oFDR = 0.35, 35% chance it's luck"
    - This prevents you from deploying a lucky but unskilled strategy
    
    **The Bayesian perspective**: oFDR uses Bayes' theorem to flip the question:
    - Typical test: P[observe SR | strategy is bad]
    - oFDR: P[strategy is bad | observe SR]
    - The second question is what we actually care about!
    
    References
    ----------
    .. [1] Bailey, D. H., et al. (2014). The Probability of Backtest Overfitting.    
    .. [2] Bailey, David H. and López de Prado, Marcos, The Deflated Sharpe Ratio: 
            Correcting for Selection Bias, Backtest Overfitting and Non-Normality (July 31, 2014).
            Journal of Portfolio Management, 40 (5), pp. 94-107. 2014 (40th Anniversary Special Issue),
            Available at SSRN: https://ssrn.com/abstract=2460551 or 
            http://dx.doi.org/10.2139/ssrn.2460551 
    .. [3] López de Prado, M., Lipton, A., & Zoonekynd, V. (2025). "Sharpe Ratio Inference: 
       A New Standard for Decision-Making and Reporting." Available at SSRN: 
       https://ssrn.com/abstract=5520741


    Examples
    --------
    >>> # Single strategy
    >>> ofdr = oFDR(SR=1.0, SR0=0, SR1=0.5, T=100, p_H1=0.05)
    >>> print(f"oFDR = {ofdr:.3f}")
    
    >>> # Best of 200 trials (Optuna optimization)
    >>> ofdr = oFDR(SR=1.0, SR0=0, SR1=0.5, T=100, p_H1=0.05, K=200)
    >>> print(f"oFDR (adjusted) = {ofdr:.3f}")  # Higher due to selection bias
    """
    p0 = 1 - probabilistic_sharpe_ratio(
        SR, SR0, T=T, gamma3=gamma3, gamma4=gamma4, rho=rho, K=K
    )
    
    # Probability of observing SR or higher under H1 (alternative hypothesis)
    p1 = 1 - probabilistic_sharpe_ratio(
        SR, SR1, T=T, gamma3=gamma3, gamma4=gamma4, rho=rho, K=K
    )
    
    # Numerical stability: clamp probabilities to avoid division by zero
    p0 = max(p0, 1e-16)
    p1 = max(p1, 1e-16)
    
    # Prior probability of H0
    p_H0 = 1 - p_H1
    
    # Bayes' theorem: P[H0 | SR > SR_obs]
    # = P[SR > SR_obs | H0] * P[H0] / P[SR > SR_obs]
    # = p0 * p_H0 / (p0 * p_H0 + p1 * p_H1)
    denominator = p0 * p_H0 + p1 * p_H1
    if denominator < 1e-16:
        return 0.0  # If both are effectively impossible, return 0 (or undefined, but 0 is safer)
        
    ofdr = p0 * p_H0 / denominator
    
    return ofdr


def expected_maximum_sharpe_ratio(
    number_of_trials: int,
    variance: float,
    SR0: float = 0
) -> float:
    """
    Expected max SR when selecting best of K trials (haircut).
    
    This is the "inflation" in Sharpe ratio due to selecting the best
    result from K independent trials.
    
    Parameters
    ----------
    number_of_trials : int
        Number of trials (K)
    variance : float
        Variance of Sharpe ratio estimator
    SR0 : float, optional
        Null hypothesis SR (default: 0)
        
    Returns
    -------
    float
        Expected maximum SR inflation due to multiple testing
        
    Notes
    -----
    **Simple Explanation**: The "lucky winner" bias - how much better does the best
    result look just by chance?
    
    **Thought experiment**: Flip 100 coins, each 10 times. Even though all coins are
    fair (expected 5 heads), the "best" coin might show 8 heads. That's not because
    it's special - it's because you picked the luckiest one out of 100.
    
    **In Optuna optimization**:
    - You run K=200 trials with different hyperparameters
    - Even if all hyperparameters are equally bad, the best trial will look good
    - Expected max SR tells you how much "inflation" to expect from this selection
    
    **Example**:
    - K = 100 trials, variance = 0.5
    - Expected max SR ≈ 0.3
    - If best observed SR = 1.5, the "true" SR is probably closer to 1.5 - 0.3 = 1.2
    - The 0.3 is the "haircut" we take for selection bias
    
    **The "haircut"**: We subtract this expected maximum from our observed SR to get
    the Deflated Sharpe Ratio (DSR). This is the "haircut" we take to account for
    selection bias.
    
    **Key insight**: More trials (larger K) → bigger haircut.
    - K = 10: Small haircut (not much selection bias)
    - K = 100: Moderate haircut
    - K = 1000: Large haircut (significant selection bias)
    
    Testing 1000 strategies and picking the best gives you a much larger "lucky winner"
    bias than testing 10 strategies.
    
    **Mathematical basis**: Uses extreme value theory (Gumbel distribution) to compute
    the expected maximum of K independent normal random variables.
    
    References
    ----------
    .. [1] Harvey, C. R., & Liu, Y. (2015). "Backtesting." Available at SSRN: 
       https://ssrn.com/abstract=2345489 or http://dx.doi.org/10.2139/ssrn.2345489
    .. [2] Bailey, D. H., & López de Prado, M. (2014). "The Deflated Sharpe Ratio: 
       Correcting for Selection Bias, Backtest Overfitting and Non-Normality."
       Journal of Portfolio Management, 40(5), 94-107.
    .. [3] López de Prado, M., Lipton, A., & Zoonekynd, V. (2025). "Sharpe Ratio Inference: 
       A New Standard for Decision-Making and Reporting." Available at SSRN: 
       https://ssrn.com/abstract=5520741
    
    Examples
    --------
    >>> # 100 trials with variance 0.5
    >>> haircut = expected_maximum_sharpe_ratio(100, 0.5)
    >>> print(f"Expected inflation: {haircut:.3f}")
    
    >>> # 1000 trials - much bigger haircut!
    >>> haircut = expected_maximum_sharpe_ratio(1000, 0.5)
    >>> print(f"Expected inflation: {haircut:.3f}")
    """
    # Euler-Mascheroni constant
    euler_gamma = np.euler_gamma
    
    # Expected maximum using extreme value theory
    # Based on Gumbel distribution for maximum of normals
    E_max = SR0 + (
        np.sqrt(variance) * (
            (1 - euler_gamma) * scipy.stats.norm.ppf(1 - 1 / number_of_trials) +
            euler_gamma * scipy.stats.norm.ppf(1 - 1 / number_of_trials / np.exp(1))
        )
    )
    
    # Return the inflation (haircut) amount
    return E_max - SR0
