import math
import warnings
from typing import Any, Dict, Tuple, Union

import numpy as np
import numpy.typing as npt
import scipy
from scipy import stats

np.seterr(all="ignore")  # ignore floating point warnings

VALID_ALTERNATIVES = ("two_sided", "greater", "less")
VALID_TEST_TYPES = ("one_sample", "paired", "two_sample")


def check_alternative(alternative: str) -> None:
    """Helper function that raises an error in case the alternative hypothesis is invalid"""
    if alternative not in VALID_ALTERNATIVES:
        raise ValueError(
            f"alternative must be equal to 'two-sided', 'less', or 'greater', got: {alternative}"
        )


def check_test_type(test_type: str) -> None:
    """Helper function that raises an error in case the test type is invalid"""
    if test_type not in VALID_TEST_TYPES:
        raise ValueError(
            f"test type must be equal to 'two_sample', 'one_sample', or 'paired', got: {test_type}"
        )


def define_ttest_n(
    low_n: int = 3, high_n: int = 100, ratio: float = 1, test_type: str = "one_sample"
) -> Dict[str, Any]:
    """Helper function that outputs the sample sizes, effective sample sizes and the degrees of
    freedom depending on the type of t-test.

    Parameters:
        low_n (float): minimal sample size of the (first) sample when computing the power due to
    optional stopping. Default low_n is set 1.
        high_n (float): minimal sample size of the (first) sample when computing the power due to
    optional stopping. Default high_n is set 1000.
        ratio (float): > 0; the randomisation ratio of condition 2 over condition 1. If test_type
    is not equal to "two_sample", or if n_plan is of len(1) then ratio=1.
        test_type (float): one of "one_sample", "paired", "two_sample".

    Returns:
        result (dict): containing the sample sizes, effective sample sizes, and the degrees of
    freedom for the type of t-test.

    """
    check_test_type(test_type)

    n1 = np.arange(low_n, high_n + 1)

    if test_type == "two_sample":
        n2 = np.ceil(ratio * n1)
        n_eff = np.ceil(n1 * ratio / (1 + ratio)).astype(int)
        nu = (1 + ratio) * n1 - 2
    else:
        n2 = np.zeros(shape=n1.shape)
        n_eff = np.ceil(n1).astype(int)
        nu = n_eff - 1

    result = {"n1": n1, "n2": n2, "n_eff": n_eff, "nu": nu}
    return result


def safe_ttest_stat(
    t: Union[float, npt.NDArray[np.float64]],
    parameter: Union[float, npt.NDArray[np.float64]],
    n1: Union[int, npt.NDArray[np.int64]],
    n2: Union[int, npt.NDArray[np.int64]] = 0,
    alternative: str = "two_sided",
    t_density: bool = False,
    paired: bool = False,
) -> Any:
    """Computes E-Values Based on the T-Statistic; A summary stats version of safe_ttest with the
    data replaced by t, n1 and n2, and the design object by delta_s.

    Parameters:
        t (float): that represents the observed t-statistic.
        parameter (float): this defines the safe test S, i.e., a likelihood ratio of t
    distributions with in the denominator the likelihood with delta = 0 and in the numerator an
    average likelihood defined by 1/2 time the likelihood at the non-centrality parameter
    sqrt(n_eff)*parameter and 1/2 times the likelihood at the non-centrality parameter
    -sqrt(n_eff)*parameter.
        n1 (int): that represents the size in a one-sample t-test, (n2=0). When n2 is not 0,
    this specifies the size of the first sample for a two-sample test.
        n2 (int): optional integer that specifies the size of the second sample. If it's left
    unspecified, thus, 0 it implies that the t-statistic is based on one-sample.
        t_density (bool) the representation of the safe t-test as the likelihood ratio of t
    densities.
        paired (bool) whether the statistic is for a paired t-test

    Returns:
        result (float): the e-value in favour of the alternative over the null, corresponding to
    the t-value

    """

    check_alternative(alternative)
    delta_s = parameter
    if isinstance(n1, (int, float, np.int64)):
        n1 = np.array([n1])
    if isinstance(n2, (int, float, np.int64)):
        n2 = np.array([n2])

    if all(n2 == 0) or paired:
        n_eff = n1.astype(float)
        nu = n1 - 1
    else:
        n_eff = 1 / (1 / n1 + 1 / n2)
        nu = n1 + n2 - 2

    a = t**2 / (nu + t**2)
    exp_term = np.exp((a - 1) * n_eff * delta_s**2 / 2)
    z_arg = (-1) * a * n_eff * delta_s**2 / 2

    a_kummer = scipy.special.hyp1f1(-nu / 2, 1 / 2, z_arg)

    if alternative == "two_sided":
        result = (exp_term * a_kummer).astype(float)
    else:
        b_kummer = np.nan_to_num(
            np.exp(scipy.special.loggamma(nu / 2 + 1) - scipy.special.loggamma((nu + 1) / 2))
            * np.sqrt(2 * n_eff)
            * delta_s
            * t
            / np.sqrt(t**2 + nu)
            * scipy.special.hyp1f1((1 - nu) / 2, 3 / 2, z_arg),
            nan=np.inf,
        )
        result = (exp_term * (a_kummer + b_kummer)).astype(float)

    if len(n1) == 1 and result < 0:
        warnings.warn("Numerical overflow: e_value close to zero. Ratio of t density employed.")
        result = safe_ttest_stat_t_density(
            t=t, parameter=parameter, nu=nu, n_eff=n_eff, alternative=alternative
        )

    # if len(n1) == 1 and result == np.inf:
    #     return 0

    return result


def safe_ttest_stat_t_density(
    t: Union[float, npt.NDArray[np.float64]],
    parameter: Union[float, npt.NDArray[np.float64]],
    nu: Union[float, int, npt.NDArray[Any]],
    n_eff: Union[float, int, npt.NDArray[np.float64]],
    alternative: str = "two_sided",
    paired: bool = False,
) -> Any:
    check_alternative(alternative)

    delta_s = parameter

    if alternative == "twoSided":
        log_term1 = stats.nct.logpdf(t, df=nu, nc=np.sqrt(n_eff) * delta_s) - stats.nct.logpdf(
            t, df=nu, nc=0
        )
        log_term2 = stats.nct.logpdf(t, df=nu, nc=-np.sqrt(n_eff) * delta_s) - stats.nct.logpdf(
            t, df=nu, nc=0
        )

        result = np.exp(log_term1 + log_term2) / 2
    else:
        result = stats.nct.pdf(t, df=nu, nc=np.sqrt(n_eff) * delta_s) / stats.nct.pdf(
            t, df=nu, nc=0
        )

    if result < 0:
        warnings.warn("Numerical overflow: E-value is essentially zero")
        return 1e-9

    return result


def safe_ttest_stat_alpha(
    t: float,
    parameter: float,
    alpha: float,
    n1: int,
    n2: int = 0,
    alternative: str = "two_sided",
    t_density: bool = False,
) -> Any:
    check_alternative(alternative)

    return (
        safe_ttest_stat(
            t=t, parameter=parameter, n1=n1, n2=n2, alternative=alternative, t_density=t_density
        )
        - 1 / alpha
    )


def get_q_beta(
    idx: int,
    delta_true: float,
    beta: float,
    alternative: str,
    nu_vector: npt.NDArray[Any],
    n_eff_vector: npt.NDArray[Any],
) -> Any:
    if alternative == "two_sided":
        q_beta = np.sqrt(
            stats.ncf.ppf(
                q=beta, dfn=1, dfd=nu_vector[idx], nc=n_eff_vector[idx] * delta_true**2
            )
        )

    else:
        q_beta = stats.nct.ppf(
            q=beta, df=nu_vector[idx], nc=np.sqrt(n_eff_vector[idx]) * delta_true
        )
    return q_beta


def nplan_binary_search(
    n_def: Dict[str, Any],
    delta_true: float,
    delta_min: float,
    low_n: int,
    high_n: int,
    alpha: float = 0.05,
    beta: float = 0.2,
    ratio: float = 1,
    test_type: str = "one_sample",
    alt: str = "two_sided",
) -> int:
    n1_vector = n_def["n1"]
    n2_vector = n_def["n2"]
    nu_vector = n_def["nu"]
    n_eff_vector = n_def["n_eff"]

    mid = int((high_n + low_n) / 2)

    if high_n >= low_n:
        q_beta = get_q_beta(
            idx=mid,
            delta_true=delta_true,
            beta=beta,
            alternative=alt,
            nu_vector=nu_vector,
            n_eff_vector=n_eff_vector,
        )

        e_value = safe_ttest_stat(
            t=q_beta,
            parameter=delta_min,
            n1=n1_vector[mid],
            n2=n2_vector[mid],
            alternative=alt,
        )

        if e_value > 1 / alpha:
            return nplan_binary_search(
                n_def=n_def,
                delta_true=delta_true,
                delta_min=delta_min,
                low_n=low_n,
                high_n=mid - 1,
                alpha=alpha,
                beta=beta,
                ratio=ratio,
                test_type=test_type,
                alt=alt,
            )

        else:
            # e_value <= 1 / alpha
            return nplan_binary_search(
                n_def=n_def,
                delta_true=delta_true,
                delta_min=delta_min,
                low_n=mid + 1,
                high_n=high_n,
                alpha=alpha,
                beta=beta,
                ratio=ratio,
                test_type=test_type,
                alt=alt,
            )
    # elif low_n == len(n1_vector):
    #     return -1
    else:
        # Element is not present in the array
        return mid + 1


def compute_n_plan_batch(
    delta_min: float,
    alpha: float = 0.05,
    beta: float = 0.2,
    alternative: str = "two_sided",
    test_type: str = "one_sample",
    low_n: int = 3,
    high_n: int = 10000,
    ratio: float = 1,
) -> Dict[str, Any]:
    """Helper function: Computes the planned sample size for the safe t-test based on the minimal
    clinically relevant standardised effect size, alpha and beta.

    Parameters:
        delta_min (float): the minimal relevant standardised effect size, the smallest effect size
    that we would the experiment to be able to detect.
        alpha (float): in (0, 1); specifies the tolerable type I error control --independent of n--
    that the designed test has to adhere to. Note that it also defines the rejection rule e10 >
    1/alpha.
        beta (float): in (0, 1); specifies the tolerable type II error control necessary to
    calculate both the sample sizes and delta_s, which defines the test. Note that 1-beta defines
    the power.
        alternative (float): character string specifying the alternative hypothesis must be one of
    "two_sided" (default), "greater" or "less".
        test_type (float): one of "one_sample", "paired", "two_sample".
        low_n (float): minimal sample size of the (first) sample when computing the power due to
    optional stopping. Default low_n is set 1.
        high_n (float): minimal sample size of the (first) sample when computing the power due to
    optional stopping. Default high_n is set 1000.
        ratio (float): > 0; the randomisation ratio of condition 2 over condition 1. If test_type
    is not equal to "two_sample", or if n_plan is of len(1) then ratio=1.

    Returns:
        result (dict): containing
            n_plan: the sample size for the safe t-test
            delta_s: the minimal relevant effect size for the specified alternative

    """

    check_alternative(alternative)
    check_test_type(test_type)

    delta_min = abs(delta_min)
    delta_true = delta_min
    delta_s = delta_min

    result: Dict[str, Any] = {"delta_s": delta_s}
    n_plan: Union[int, Tuple[int, int]] = 0

    n1_plan: int = 0
    n2_plan: int = 0

    if alternative == "less":
        alt = "greater"
        result["delta_s"] = -delta_s
    else:
        alt = alternative

    idx = -1
    while idx < 0:
        n_def = define_ttest_n(low_n=low_n, high_n=high_n, ratio=ratio, test_type=test_type)

        q_beta_high = get_q_beta(
            idx=int(high_n - low_n),
            delta_true=delta_true,
            beta=beta,
            alternative=alt,
            nu_vector=n_def["nu"],
            n_eff_vector=n_def["n_eff"],
        )

        e_value = safe_ttest_stat(
            t=q_beta_high,
            parameter=delta_min,
            n1=int(high_n - low_n),
            n2=int(high_n - low_n),
            alternative=alt,
        )
        if e_value == np.inf:
            high_n = high_n // 3
        elif e_value > 20:
            idx = nplan_binary_search(
                n_def=n_def,
                delta_true=delta_true,
                delta_min=delta_min,
                low_n=0,
                high_n=high_n - low_n,
                alpha=alpha,
                beta=beta,
                ratio=ratio,
                test_type=test_type,
                alt=alt,
            )
            if idx == len(n_def["n1"]):
                idx -= 1
            n1_plan = math.ceil(n_def["n1"][idx])
            n2_plan = math.ceil(n_def["n2"][idx])
        else:
            low_n = high_n
            high_n = 5 * high_n

        if high_n > 3e8:
            n1_plan = int(1e8)
            n2_plan = int(1e8)
            break

    # idx = min(idx, len(n_def['n1'])-1)
    # if idx < 0:
    #     result["n_plan"] = (idx, idx)
    #     return result

    if not n1_plan:
        raise ValueError(
            f"Could not compute a batch sample size. Increase low_n and high_n, \
                         which are now {low_n} and {high_n}, respectively."
        )

    if test_type == "paired":
        n2_plan = n1_plan

    if not n2_plan:
        n_plan = n1_plan
    else:
        n_plan = (n1_plan, n2_plan)

    result["n_plan"] = n_plan
    return result


def generate_normal_data(
    n_plan: Union[int, Tuple[int, int]],
    n_sim: int = 1000,
    delta_true: float = 0.0,
    mu_global: float = 0.0,
    sigma_true: float = 1.0,
    paired: bool = False,
    seed: int = 0,
    mu_true: float = 0.0,
) -> Dict[str, Any]:
    if (isinstance(n_plan, int) and n_plan <= 0) or (
        isinstance(n_plan, tuple) and not all([n > 0 for n in n_plan])
    ):
        raise ValueError(f"Invalid value for n_plan: {n_plan}")

    # if bool(delta_true) + bool(mu_true) != 1:
    #     raise ValueError("Please provide either delta_true (t-test), or mu_true (z-test).")

    np.random.seed(seed)

    if not mu_true:
        mu_true = delta_true * sigma_true

    if isinstance(n_plan, int):
        data_group1 = np.random.normal(loc=mu_true, scale=sigma_true, size=n_plan * n_sim)
        data_group1 = data_group1.reshape((n_sim, n_plan))
        data_group2 = np.zeros(shape=(n_sim, n_plan))
    else:
        n1_plan, n2_plan = n_plan[0], n_plan[1]

        if paired:
            coef = 1 / np.sqrt(2)
        else:
            coef = 1 / 2

        data_group1 = np.random.normal(
            size=int(n1_plan * n_sim), loc=mu_global + coef * mu_true, scale=sigma_true
        )
        data_group1 = data_group1.reshape((n_sim, n1_plan))
        data_group2 = np.random.normal(
            size=int(n2_plan * n_sim), loc=mu_global - coef * mu_true, scale=sigma_true
        )
        data_group2 = data_group2.reshape((n_sim, n2_plan))

    return {"data_group1": data_group1, "data_group2": data_group2}


def check_and_return_parameter(
    param_to_check: float,
    alternative: str = "two_sided",
    es_min_name: str = "no_name",
    param_domain: str = "",
) -> float:
    """Checks consistency between the sided of the hypothesis and the  minimal clinically relevant
    effect size or safe test defining parameter. Throws an error if the hypothesis is incongruent
    with the parameter

    Parameters:
        param_to_check: (float) Either a named safe test defining parameter such as phi_s, or
    theta_s, or a minimal clinically relevant effect size called with a non-null es_min_name name
        alternative (float): character string specifying the alternative hypothesis must be one of
        "two_sided" (default), "greater" or "less".
        es_min_name: (string) the name of the effect size. Either "mean_diff_min" for the z-test,
    "delta_min" for the t-test, or "hr_min" for the logrank test
        param_domain: (string) typically positive_numbers. Default None

    Returns:
        param_to_check (float) after checking, perhaps with a change in sign

    """
    assert es_min_name in (
        "no_name",
        "mean_diff_min",
        "phi_s",
        "delta_min",
        "delta_s",
        "hr_min",
        "theta_s",
        "delta_true",
    )

    if alternative == "two_sided":
        if es_min_name in ("mean_diff_min", "delta_min", "delta_true"):
            return abs(param_to_check)

        return param_to_check

    if es_min_name == "no_name":
        param_name = None
    else:
        param_name = es_min_name

    if not param_name:
        param_name = "the safe test defining parameter"
        hypparam_name = "test relevant parameter"
        param_domain = "unknown"
    elif param_name == "phi_s" or es_min_name == "mean_diff_min":
        hypparam_name = "meanDiff"
        param_domain = "real_numbers"
    elif param_name == "delta_s" or es_min_name == "delta_min" or es_min_name == "delta_true":
        hypparam_name = "delta"
        param_domain = "real_numbers"
    elif param_name == "theta_s" or es_min_name == "hr_min":
        hypparam_name = "theta"
        param_domain = "positive_numbers"
    else:
        hypparam_name = "testRelevantParameter"

    if param_domain == "unknown":
        if alternative == "greater" and param_to_check < 0:
            warnings.warn(
                'The safe test defining parameter is incongruent with alternative "greater". '
                + "This safe test parameter is made positive to compare H+: "
                + "test-relevant parameter > 0 against H0 : test-relevant parameter = 0",
            )
            param_to_check = -param_to_check

        if alternative == "less" and param_to_check > 0:
            warnings.warn(
                'The safe test defining parameter is incongruent with alternative "less". '
                + "This safe test parameter is made positive to compare H-: "
                + "test-relevant parameter < 0 against H0 : test-relevant parameter = 0",
            )
            param_to_check = -param_to_check

    elif param_domain == "real_numbers":
        if alternative == "greater" and param_to_check < 0:
            warnings.warn(
                f"{param_name} incongruent with alternative 'greater'\
                  {param_name} set to - {param_name} > 0 in order to compare H+:\
                  {hypparam_name} > 0 against H0 : {hypparam_name} = 0"
            )
            param_to_check = -param_to_check

        if alternative == "less" and param_to_check > 0:
            warnings.warn(
                f"{param_name} incongruent with alternative 'greater'\
                  {param_name} set to - {param_name} < 0 in order to compare H-:\
                  {hypparam_name} < 0 against H0 : {hypparam_name} = 0"
            )
            param_to_check = -param_to_check
    elif param_domain == "positive_numbers":
        if alternative == "greater" and param_to_check < 1:
            warnings.warn(
                f"{param_name} incongruent with alternative 'greater'\
                  {param_name} set to 1/ {param_name} > 1 in order to compare H+:\
                  {hypparam_name} > 1 against H0 : {hypparam_name} = 1"
            )

            param_to_check = 1 / param_to_check

        if alternative == "less" and param_to_check > 1:
            warnings.warn(
                f"{param_name} incongruent with alternative 'greater'\
                  {param_name} set to 1/ {param_name} < 1 in order to compare H-:\
                  {hypparam_name} < 1 against H0 : {hypparam_name} = 1"
            )
            param_to_check = 1 / param_to_check

    return param_to_check


def determine_stopping(
    x1: npt.NDArray[np.float64],
    x2: npt.NDArray[np.float64],
    ndef: Dict[str, Any],
    delta_s: float,
    test_type: str = "two_sample",
    alternative: str = "two_sided",
    alpha: float = 0.05,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    n1_vector = ndef["n1"]
    n2_vector = ndef["n2"]
    n_eff_vector = ndef["n_eff"]
    paired = test_type in ("one_sample", "paired")

    if paired:
        x1_bar_vector = 1 / n1_vector * np.cumsum(x1)
        x1_square_vector = np.cumsum(x1**2)
        n1_vector = np.where(n1_vector <= 1, 1, n1_vector)
        sx1_vector = np.sqrt(
            1 / (n1_vector - 1) * (x1_square_vector - n1_vector * x1_bar_vector**2)
        )

        t_values = np.sqrt(n_eff_vector) * x1_bar_vector / sx1_vector
    else:
        x1_bar_vector = np.cumsum(x1) / n1_vector
        x1_square_vector = np.cumsum(x1**2)

        x2_bar_vector = np.cumsum(x2) / n2_vector
        x2_square_vector = np.cumsum(x2**2)

        sp_vector = np.sqrt(
            1
            / (n1_vector + n2_vector - 2)
            * (
                x1_square_vector
                - n1_vector * x1_bar_vector**2
                + x2_square_vector
                - n2_vector * x2_bar_vector**2
            )
        )
        sp_vector = np.where(n1_vector + n2_vector - 2 <= 0, 1, sp_vector)

        t_values = np.sqrt(n_eff_vector) * (x1_bar_vector - x2_bar_vector) / sp_vector

    evidence_now = safe_ttest_stat(
        t=t_values,
        parameter=delta_s,
        n1=n1_vector,
        n2=n2_vector,
        alternative=alternative,
        paired=paired,
    )

    idx_arr = np.where(evidence_now > 1 / alpha)[0]

    if len(idx_arr) == 0:
        idx = -3
    else:
        idx = int(idx_arr[0])
    e_value = evidence_now[idx]

    stopping_time = n1_vector[idx]
    e_value_stop = e_value

    return stopping_time, e_value_stop


def sample_stopping_times(
    delta_true: float,
    alpha: float = 0.05,
    alternative: str = "two_sided",
    test_type: str = "one_sample",
    n_sim: int = 1000,
    n_max: Union[int, Tuple[int, int]] = 1000,
    ratio: float = 1.0,
    parameter: float = 0.0,
    seed: int = 0,
    sigma_true: float = 1.0,
    paired: bool = False,
    sim_no_eff: bool = False,
) -> Dict[str, npt.NDArray[np.float64]]:
    """Simulate stopping times for the safe t-test

    Parameters:
        delta_true (float): the value of the true standardised effect size (test-relevant
    parameter). This argument is used by `design_safe_t()` with `delta_true = delta_min`
        n_max (int): > 0; maximum sample size of the (first) sample in each sample path.
        want_evalues_at_n_max (bool). If True then compute e_values at n_max. Default False.
        alpha (float): in (0, 1); specifies the tolerable type I error control --independent of n--
    that the designed test has to adhere to. Note that it also defines the rejection rule e10 >
    1/alpha.
        alternative (float): character string specifying the alternative hypothesis must be one of
        "two_sided" (default), "greater" or "less".
        low_n (float): minimal sample size of the (first) sample when computing the power due to
    optional stopping. Default low_n is set 1.
        test_type (float): one of "one_sample", "paired", "two_sample".
        ratio (float): > 0; the randomisation ratio of condition 2 over condition 1. If test_type
    is not equal to "two_sample", or if n_plan is of len(1) then ratio=1.
        parameter (float): test defining parameter. Default set to 0.
        n_sim (float): > 0; the number of simulations needed to compute power or the number of
    samples paths for the safe z test under continuous monitoring.
        seed (float): seed number.

    Returns:
        result (dict): with stopping_times and break_vector. Entries of break_vector are 0, 1. A 1
    represents stopping due to exceeding n_max, and 0 due to 1/alpha threshold crossing, which
    implies that in corresponding stopping time is infinite.
    """
    if alpha <= 0 or alpha > 1:
        raise ValueError(f"Invalid value for alpha: {alpha} or n_max: {n_max}")

    # Object that will be returned. A sample of stopping times
    stopping_times, break_vector = np.zeros(shape=n_sim), np.zeros(shape=n_sim)
    e_values_stopped, evalues_at_n_max = np.zeros(shape=n_sim), np.zeros(shape=n_sim)

    if parameter:
        delta_s = parameter
    elif sim_no_eff:
        delta_s = 0
    else:
        delta_true = check_and_return_parameter(
            delta_true, alternative=alternative, es_min_name="delta_true"
        )
        delta_s = delta_true

    if isinstance(n_max, int):
        n_max = (n_max, math.ceil(ratio * n_max))

    if test_type == "two_sample":
        n1_max = n_max[0]
        ratio = n_max[1] / n_max[0]
    elif test_type in ("paired", "one_sample"):
        n1_max = n_max[0]
        n_max = n1_max
        ratio = 1

    low_n = 3
    ndef = define_ttest_n(low_n=low_n, high_n=n1_max + low_n - 1, ratio=ratio, test_type=test_type)

    sim_data = generate_normal_data(
        n_plan=n_max,
        n_sim=n_sim,
        delta_true=delta_s,
        sigma_true=sigma_true,
        paired=paired,
        seed=seed,
    )

    for sim in range(len(stopping_times)):
        x1 = sim_data["data_group1"][sim]
        x2 = sim_data["data_group2"][sim]

        stop_time, e_value_stop = determine_stopping(
            x1=x1,
            x2=x2,
            ndef=ndef,
            delta_s=delta_true,
            test_type=test_type,
            alternative=alternative,
        )

        stopping_times[sim] = stop_time
        e_values_stopped[sim] = e_value_stop

        if stop_time >= n1_max:
            break_vector[sim] = 1

    result = {
        "stopping_times": stopping_times,
        "break_vector": break_vector,
        "e_values_stopped": e_values_stopped,
        "evalues_at_n_max": evalues_at_n_max,
    }

    return result


def compute_boot_obj(
    values: npt.NDArray[np.float64],
    beta: float = 0.0,
    n_plan: Union[int, Tuple[int, int]] = 0,
    n_boot: int = 1000,
    alpha: float = 0.0,
    obj_type: str = "n_plan",
    seed: int = 0,
) -> Dict[str, Any]:
    """Computes the bootstrap object for sequential sampling procedures regarding n_plan,
    beta, the implied target

    Parameters:
        values (ndarray): If obj_type equals "n_plan" or "beta" then values should be stopping
    times, if obj_type equals "log_implied_target" then values should be e_values.
        alpha (float): in (0, 1); specifies the tolerable type I error control --independent of n--
    that the designed test has to adhere to. Note that it also defines the rejection rule e10 >
    1/alpha.
        beta (float): in (0, 1); specifies the tolerable type II error control necessary to
    calculate both the sample sizes and delta_s, which defines the test. Note that 1-beta defines
    the power.
        n_plan (int): > 0; representing the number of planned samples (for the first group).
        obj_type (string): either "n_plan", "n_mean", "beta", "beta_from_e_values",
    "expected_stop_time" or "log_implied_target".
        n_boot (float): > 0; representing the number of bootstrap samples to assess the accuracy of
    approximation of the power, the number of samples for the safe z test under continuous
    monitoring, or for the computation of the logarithm of the implied target.

    Returns:
    boot_obj: (dict) containing the following values
        to: the statistic for the provided sample
        t: the statistic of bootstrapped samples
        bootsSe: the standard deviation of the bootstrapped samples

    """
    np.random.seed(seed)
    if not isinstance(n_plan, int):
        n_plan = n_plan[0]

    valid_obj_types = [
        "n_plan",
        "n_mean",
        "beta",
        "beta_from_e_values",
        "log_implied_target",
        "expected_stop_time",
    ]

    if obj_type not in valid_obj_types:
        raise ValueError(f"obj_type must be one of: {valid_obj_types}")

    def f_beta(x: npt.NDArray[np.float64]) -> np.float64:
        return 1 - np.mean(x <= n_plan)

    def f_beta_from_e_values(x: npt.NDArray[np.float64]) -> np.float64:
        return np.mean(x >= 1 / alpha)

    def f_n_plan(x: npt.NDArray[np.float64]) -> np.float64:
        return np.quantile(x, q=1 - beta)

    def f_n_mean(x: npt.NDArray[np.float64]) -> np.float64:
        x = np.where(x > n_plan, n_plan, x)
        return np.mean(x)

    def f_log_implied_target(x: npt.NDArray[np.float64]) -> Any:
        return np.mean(np.log(x))

    def f_expected_stop_time(x: npt.NDArray[np.float64]) -> np.float64:
        return np.mean(x)

    functions = {
        "beta": f_beta,
        "beta_from_e_values": f_beta_from_e_values,
        "n_plan": f_n_plan,
        "n_mean": f_n_mean,
        "log_implied_target": f_log_implied_target,
        "expected_stop_time": f_expected_stop_time,
    }

    f = functions[obj_type]

    ts = list()
    for _ in range(n_boot):
        s = np.random.choice(values, size=len(values), replace=True)
        t = f(s)
        ts.append(t)

    t0 = f(values)

    boot_obj = {"t0": t0, "t": ts, "boot_se": np.std(ts, ddof=1)}
    return boot_obj


def compute_n_plan(
    delta_min: float,
    beta: float = 0.2,
    alpha: float = 0.05,
    alternative: str = "two_sided",
    test_type: str = "one_sample",
    low_n: int = 3,
    high_n: int = 10000,
    ratio: float = 1.0,
    n_sim: int = 1000,
    n_boot: int = 1000,
    parameter: float = 0.0,
    seed: int = 0,
) -> Dict[str, Any]:
    """Helper function: Computes the planned sample size of the safe t-test based on the
    minimal clinical relevant standardised mean difference.

    Parameters:
        delta_min (float): the minimal relevant standardised effect size, the smallest effect size
    that we would the experiment to be able to detect.
        alpha (float): in (0, 1); specifies the tolerable type I error control --independent of n--
    that the designed test has to adhere to. Note that it also defines the rejection rule e10 >
    1/alpha.
        beta (float): in (0, 1); specifies the tolerable type II error control necessary to
    calculate both the sample sizes and delta_s, which defines the test. Note that 1-beta defines
    the power.
        alternative (float): character string specifying the alternative hypothesis must be one of
    "two_sided" (default), "greater" or "less".
        low_n (float): minimal sample size of the (first) sample when computing the power due to
    optional stopping. Default low_n is set 1.
        high_n (float): minimal sample size of the (first) sample when computing the power due to
    optional stopping. Default high_n is set 1000.
        test_type (float): one of "one_sample", "paired", "two_sample".
        ratio (float): > 0; the randomisation ratio of condition 2 over condition 1. If test_type
    is not equal to "two_sample", or if n_plan is of len(1) then ratio=1.
        parameter (float): test defining parameter. Default set to 0.
        n_sim (float): > 0; the number of simulations needed to compute power or the number of
    samples paths for the safe z test under continuous monitoring.
        n_boot (float): > 0; representing the number of bootstrap samples to assess the accuracy of
    approximation of the power, the number of samples for the safe z test under continuous
    monitoring, or for the computation of the logarithm of the implied target.
        seed (float): seed number.

    Returns:
        result: (dict) contains at least n_plan and an adapted bootstrap object

    """

    if parameter:
        delta_s = parameter
    else:
        delta_min = check_and_return_parameter(
            param_to_check=delta_min, alternative=alternative, es_min_name="delta_min"
        )
        delta_s = delta_min

    temp_obj = compute_n_plan_batch(
        delta_min=delta_min,
        alpha=alpha,
        beta=beta,
        alternative=alternative,
        test_type=test_type,
        low_n=low_n,
        high_n=high_n,
        ratio=ratio,
    )
    n_plan_batch = temp_obj["n_plan"]

    sampling_results = sample_stopping_times(
        delta_true=delta_min,
        alpha=alpha,
        alternative=alternative,
        seed=seed,
        n_sim=n_sim,
        n_max=n_plan_batch,
        ratio=ratio,
        test_type=test_type,
        parameter=delta_s,
    )

    times = sampling_results["stopping_times"]

    boot_obj_nplan = compute_boot_obj(values=times, obj_type="n_plan", beta=beta, n_boot=n_boot)

    n1_plan = math.ceil(boot_obj_nplan["t0"])

    boot_obj_nmean = compute_boot_obj(
        values=times, obj_type="n_mean", n_plan=n1_plan, n_boot=n_boot
    )

    n1_mean = math.ceil(boot_obj_nmean["t0"])

    result = {
        "n1_plan": n1_plan,
        "boot_obj_nplan": boot_obj_nplan,
        "n1_mean": n1_mean,
        "boot_obj_nmean": boot_obj_nmean,
        "n_plan_batch": n_plan_batch,
    }

    return result


def design_safe_t(
    delta_min: float = 0.0,
    beta: float = 0.0,
    n_plan: Union[int, Tuple[int, int]] = 0,
    alpha: float = 0.05,
    h0: float = 0.0,
    alternative: str = "two_sided",
    low_n: int = 3,
    high_n: int = 10000,
    low_param: float = 0.01,
    high_param: float = 1.5,
    tol: float = 0.01,
    test_type: str = "one_sample",
    ratio: float = 1.0,
    n_sim: int = 1000,
    n_boot: int = 1000,
    parameter: float = 0.0,
    seed: int = 0,
) -> Dict[str, Any]:
    """Designs a Safe Experiment to Test Means with a T Test
    A designed experiment requires (1) a sample size n_plan to plan for, and (2) the parameter of
    the safe test, i.e., delta_s. If n_plan is provided, then only the safe test defining parameter
    delta_s needs to determined. That resulting delta_s leads to an (approximately) most powerful
    safe test. Typically, n_plan is unknown and the user has to specify (i) a tolerable type II
    error beta, and (ii) a clinically relevant minimal population standardised effect size
    delta_min. The procedure finds the smallest n_plan for which delta_min is found with power of
    at least 1 - beta.

    Parameters:
        delta_min (float): the minimal relevant standardised effect size, the smallest effect size
    that we would the experiment to be able to detect.
        alpha (float): in (0, 1); specifies the tolerable type I error control --independent of n--
    that the designed test has to adhere to. Note that it also defines the rejection rule e10 >
    1/alpha.
        beta (float): in (0, 1); specifies the tolerable type II error control necessary to
    calculate both the sample sizes and delta_s, which defines the test. Note that 1-beta defines
    the power.
        alternative (float): character string specifying the alternative hypothesis must be one of
    "two_sided" (default), "greater" or "less".
        n_plan (float): max len 2 representing the planned sample sizes.
        h0 (float): number indicating the hypothesised true value of the mean under the null. For
    the moment h0=0.
        low_n (float): minimal sample size of the (first) sample when computing the power due to
    optional stopping. Default low_n is set 1.
        high_n (float): minimal sample size of the (first) sample when computing the power due to
    optional stopping. Default high_n is set 1000.
        low_param (float): the smallest delta of the search space for the test-defining delta_s
    for scenario 3. Currently not yet in use.
        high_param (float): the largest delta of the search space for the test-defining delta_s
    for scenario 3. Currently not yet in use.
        tol (float): the stepsizes between the low_param and high_param.
        test_type (float): one of "one_sample", "paired", "two_sample".
        ratio (float): > 0; the randomisation ratio of condition 2 over condition 1. If test_type
    is not equal to "two_sample", or if n_plan is of len(1) then ratio=1.
        parameter (float): test defining parameter. Default set to 0.
        n_sim (float): > 0; the number of simulations needed to compute power or the number of
    samples paths for the safe z test under continuous monitoring.
        n_boot (float): > 0; representing the number of bootstrap samples to assess the accuracy of
    approximation of the power, the number of samples for the safe z test under continuous
    monitoring, or for the computation of the logarithm of the implied target.
        seed (float): seed number.

    Returns:
    result: (dict) containing the following components:
        n_plan: the planned sample size(s).
        parameter: the safe test defining parameter. Here delta_s.
        es_min: the minimal clinically relevant standardised effect size provided by the user.
        alpha: the tolerable type I error provided by the user.
        beta: the tolerable type II error provided by the user.
        alternative: any of "two-sided", "greater", "less" provided by the user.
        test_type: any of "one_sample", "paired", "two_sample" provided by the user.
        paired: logical, True if "paired", False otherwise.
        h0: the specified hypothesised value of the mean or mean difference depending on
    whether it was a one-sample or a two-sample test.}
        ratio: default is 1. Different from 1, whenever test_type equals "two_sample", then it
      defin ratio between the planned randomisation of condition 2 over condition 1.}
        low_n: the smallest n of the search space for n provided by the user.
        high_n: the largest n of the search space for n provided by the user.
        low_param: the smallest delta of the search space for delta provided by the user.
        high_param: the largest delta of the search space for delta provided by the user.
        tol: the step size between low_param and high_param provided by the user.
        pilot: False (default) specified by the user to indicate that the design is not a
    pilot study.
        call: the expression with which this function is called.

    """

    if not all([alpha > 0, alpha < 1]):
        raise ValueError(f"Invalid valid for alpha : {alpha}")

    check_alternative(alternative)
    check_test_type(test_type)

    if parameter:
        parameter = check_and_return_parameter(
            param_to_check=parameter, es_min_name="delta_s", alternative=alternative
        )

    if delta_min:
        delta_min = check_and_return_parameter(
            param_to_check=delta_min, es_min_name="delta_min", alternative=alternative
        )

    paired = True if test_type == "paired" else False

    design_scenario, note = "", ""

    n_plan_two_se: Tuple[Any, Any]
    n_mean: Union[float, Tuple[float, float]]
    n_mean_two_se: Tuple[Any, Any]

    log_implied_target, log_implied_target_two_se = 0, 0
    beta_two_se = 0

    boot_obj_nplan: Dict[str, Any]
    boot_obj_nmean: Dict[str, Any]
    boot_obj_beta: Dict[str, Any] = {}
    boot_obj_log_implied_target: Dict[str, Any] = {}

    if delta_min and beta and not n_plan:
        # scenario 1a: delta + power known, calculate n_plan
        design_scenario = "1a"
        delta_s = delta_min

        temp_result = compute_n_plan(
            delta_min=delta_min,
            beta=beta,
            alpha=alpha,
            alternative=alternative,
            test_type=test_type,
            low_n=low_n,
            high_n=high_n,
            ratio=ratio,
            seed=seed,
            n_sim=n_sim,
            n_boot=n_boot,
            parameter=delta_s,
        )

        n_plan_batch = temp_result["n_plan_batch"]
        boot_obj_nplan = temp_result["boot_obj_nplan"]
        boot_obj_nmean = temp_result["boot_obj_nmean"]

        if test_type == "one_sample":
            n_plan = temp_result["n1_plan"]
            n_plan_two_se = 2 * boot_obj_nplan["boot_se"]

            n_mean = temp_result["n1_mean"]
            n_mean_two_se = 2 * boot_obj_nmean["boot_se"]

            note = (
                "If it is only possible to look at the data once, "
                + "then n_plan = "
                + str(n_plan_batch)
                + "."
            )
        elif test_type == "paired":
            n_plan = (temp_result["n1_plan"], temp_result["n1_plan"])
            n_plan_two_se_one = 2 * boot_obj_nplan["boot_se"]
            n_plan_two_se = (n_plan_two_se_one, n_plan_two_se_one)

            n_mean = (temp_result["n1_mean"], temp_result["n1_mean"])
            n_mean_two_se_one = 2 * boot_obj_nmean["boot_se"]
            n_mean_two_se = (n_mean_two_se_one, n_mean_two_se_one)

            note = (
                "If it is only possible to look at the data once, "
                + "then n1_plan = "
                + str(n_plan_batch[0])
                + " and n2_plan = "
                + str(n_plan_batch[1])
                + "."
            )
        elif test_type == "two_sample":
            n_plan = (temp_result["n1_plan"], math.ceil(ratio * temp_result["n1_plan"]))
            n_plan_two_se_one = 2 * boot_obj_nplan["boot_se"]
            n_plan_two_se = (n_plan_two_se_one, ratio * n_plan_two_se_one)

            n_mean = (temp_result["n1_mean"], ratio * temp_result["n1_mean"])
            n_mean_two_se_one = 2 * boot_obj_nmean["boot_se"]
            n_mean_two_se = (n_mean_two_se_one, ratio * n_mean_two_se_one)

            note = (
                "If it is only possible to look at the data once, "
                + "then n1_plan = "
                + str(n_plan_batch[0])
                + " and n2_plan = "
                + str(n_plan_batch[1])
                + "."
            )

    elif delta_min and not beta and not n_plan:
        design_scenario = "1b"
        delta_s = delta_min

        n_plan = 0
        beta = 0
        delta_min = delta_min
    elif not delta_min and not beta and n_plan:
        # scenario 1c: only n_plan known, can perform a pilot (no warning though)
        design_scenario = "1c"

        return design_pilot_safe_t(
            n_plan=n_plan,
            alpha=alpha,
            h0=h0,
            alternative=alternative,
            low_param=low_param,
            tol=tol,
            paired=paired,
        )
    elif delta_min and not beta and n_plan:
        # scenario 2: given effect size and n_plan, calculate power and implied target
        design_scenario = "2"
        delta_s = delta_min

        n_plan = check_and_return_n_plan(n_plan=n_plan, ratio=ratio, test_type=test_type)

        temp_result = compute_beta_safe_t(
            delta_min=delta_min,
            n_plan=n_plan,
            alpha=alpha,
            alternative=alternative,
            test_type=test_type,
            parameter=delta_s,
            n_sim=n_sim,
            n_boot=n_boot,
            seed=seed,
        )

        beta = temp_result["beta"]
        boot_obj_beta = temp_result["boot_obj_beta"]
        beta_two_se = 2 * boot_obj_beta["boot_se"]

        log_implied_target = temp_result["log_implied_target"]
        boot_obj_log_implied_target = temp_result["boot_obj_log_implied_target"]
        log_implied_target_two_se = 2 * boot_obj_log_implied_target["boot_se"]
    elif not delta_min and beta and not n_plan:
        design_scenario = "3"
        raise ValueError("Not yet implemented")

    if not design_scenario:
        raise ValueError(
            "Can't design: Please provide this function with either: \n",
            "(1.a) non-null delta_min, non-null beta and 0 n_plan, or \n",
            "(1.b) non-null delta_min, 0 beta, and 0 n_plan, or \n",
            "(1.c) 0 delta_min, 0 beta, non-null n_plan, or \n",
            "(2) non-null delta_min, 0 beta and non-null n_plan, or \n",
            "(3) 0 delta_min, non-null beta, and non-null n_plan.",
        )

    if not delta_min:
        delta_min = 0

    # if design_scenario in ["2", "3"]:
    #     n2_plan = n_plan[1]

    result = {
        "parameter": delta_s,
        "es_min": delta_min,
        "alpha": alpha,
        "alternative": alternative,
        "h0": h0,
        "test_type": test_type,
        "paired": paired,
        "ratio": ratio,
        "pilot": False,
        "n_plan": n_plan,
        "n_plan_two_se": n_plan_two_se,
        "n_plan_batch": n_plan_batch,
        "n_mean": n_mean,
        "n_mean_two_se": n_mean_two_se,
        "beta": beta,
        "beta_two_se": beta_two_se,
        "log_implied_target": log_implied_target,
        "log_implied_target_two_se": log_implied_target_two_se,
        "boot_obj_nplan": boot_obj_nplan,
        "boot_obj_beta": boot_obj_beta,
        "boot_obj_log_implied_target": boot_obj_log_implied_target,
        "boot_obj_nmean": boot_obj_nmean,
        "note": note,
    }

    return result


def safe_ttest(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64] = np.zeros(1),
    design_obj: Any = None,
    paired: bool = False,
    pilot: bool = False,
    alpha: float = 0.05,
    alternative: str = "",
    ci_value: float = 0.95,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {}

    if not design_obj and not pilot:
        raise ValueError(
            """No design given and not indicated that this is a pilot study. Run design first and \
            provide this to safe_ttest/safe.t.test, or run safe_ttest with pilot=True""",
        )

    # if design_obj:
    #     checkDoubleArgumentsdesign_object(design_obj, "alternative"=alternative, "alpha"=alpha)

    n1 = len(x)
    x = x[~np.isnan(x)]
    estimate: Union[Any, Tuple[Any, Any]]
    n: Union[int, Tuple[int, int]]

    if not y.any():
        test_type = "one_sample"
        n_eff = float(n1)
        n = n1
        n2 = 0
        nu = n - 1

        if paired:
            raise ValueError(
                "Data error: Paired analysis requested without specifying the second variable"
            )

        mean_obs = np.mean(x)
        estimate = np.mean(x)
        sd_obs = np.std(x, ddof=1)
    else:
        n2 = len(y)
        y = y[~np.isnan(y)]

        if paired:
            if n1 != n2:
                raise ValueError(
                    "Data error: Error in complete.cases(x, y): Paired analysis requested, ",
                    "but the two samples are not of the same size.",
                )

            test_type = "paired"

            n_eff = n1
            nu = n1 - 1

            mean_obs = np.mean(x - y)
            estimate = np.mean(x - y)
            sd_obs = np.std(x - y, ddof=1)
        else:
            test_type = "two_sample"

            nu = n1 + n2 - 2
            n_eff = 1 / (1 / n1 + 1 / n2)

            s_pooled_squared = ((n1 - 1) * np.var(x, ddof=1) + (n2 - 1) * np.var(y, ddof=1)) / nu

            sd_obs = np.sqrt(s_pooled_squared)

            estimate = (np.mean(x), np.mean(y))
            mean_obs = estimate[0] - estimate[1]

        n = (n1, n2)

    if pilot:
        if not alternative:
            alternative = "two_sided"
        else:
            check_alternative(alternative)

        if not alpha:
            alpha = 0.05

        design_obj = design_pilot_safe_t(
            n_plan=n, alpha=alpha, alternative=alternative, paired=paired
        )

    if design_obj["test_type"] != test_type:
        warnings.warn(
            'The test type of design_obj is "'
            + design_obj["test_type"]
            + '", whereas the data correspond to a test_type "'
            + test_type
            + '"',
        )

    alternative = design_obj["alternative"]
    h0 = design_obj["h0"]
    alpha = design_obj["alpha"]

    if not ci_value:
        ci_value = 1 - alpha

    if ci_value < 0 or ci_value > 1:
        raise ValueError(
            "Can't make a confidence sequence with ci_value < 0 or ci_value > 1, or alpha < 0 or \
                alpha > 1"
        )

    t_stat = np.sqrt(n_eff) * (mean_obs - h0) / sd_obs

    if not t_stat:
        raise ValueError("Data error: Could not compute the t-statistic")

    e_value = safe_ttest_stat(
        t=t_stat,
        parameter=design_obj["parameter"],
        n1=n1,
        n2=n2,
        alternative=alternative,
        paired=paired,
    )

    float_sd_obs = float(sd_obs)
    result["statistic"] = t_stat
    # result["parameter"] = design_obj["delta_s"]
    result["estimate"] = estimate
    result["stderr"] = sd_obs / n_eff
    # result["data_name"] = data_name
    result["design_obj"] = design_obj
    result["test_type"] = test_type
    result["n"] = n
    result["ci_value"] = ci_value

    result["conf_seq"] = compute_confint_t(
        mean_obs=mean_obs - h0,
        sd_obs=float_sd_obs,
        n_eff=n_eff,
        nu=nu,
        delta_s=design_obj["parameter"],
        ci_value=ci_value,
    )

    result["e_value"] = e_value

    return result


def compute_confint_t(
    mean_obs: float,
    sd_obs: float,
    n_eff: float,
    nu: int,
    delta_s: float,
    ci_value: float = 0.95,
    g: float = 0.0,
) -> Tuple[float, float]:
    """Helper function: Computes the safe confidence sequence for the mean in a t-test

    Parameters:
        n_eff (int) > 0: the effective sample size. For one sample test this is just n.
        nu (int) > 0: the degrees of freedom.
        mean_obs (float): the observed mean. For two sample tests this is difference of the means.
        sd_obs (float): the observed standard deviation. For a two-sample test this is the root
    of the pooled variance.
        delta_s (float) > 0: the safe test defining parameter.
        ci_value (float) is the ci_value-level of the confidence sequence. Default ci_value=0.95.
        g (float) > 0: used as the variance of the normal prior on the population delta
    Default is None in which case g=delta^2.

    Returns:
        (tuple): vector that contains the upper and lower bound of the safe confidence sequence

    """
    if not g:
        g = delta_s**2

    trivial_confint = (-np.inf, np.inf)

    if nu <= 0:
        return trivial_confint

    alpha = 1 - ci_value

    numerator_w = nu * (np.power((1 + n_eff * g) / alpha**2, 1 / (nu + 1)) - 1)
    denominator_w = 1 - np.power((1 + n_eff * g) / alpha**2, 1 / (nu + 1)) / (1 + n_eff * g)

    w = numerator_w / denominator_w

    if w < 0:
        return trivial_confint

    shift = sd_obs / np.sqrt(n_eff) * np.sqrt(w)

    lower_cs = mean_obs - shift
    upper_cs = mean_obs + shift

    return (lower_cs, upper_cs)


def compute_beta_safe_t(
    delta_min: float,
    n_plan: Union[int, Tuple[int, int]],
    alpha: float = 0.05,
    alternative: str = "two_sided",
    test_type: str = "one_sample",
    seed: int = 0,
    parameter: float = 0.0,
    n_sim: int = 1000,
    n_boot: int = 1000,
) -> Dict[str, Any]:
    check_alternative(alternative)
    check_test_type(test_type)

    ratio = n_plan[1] / n_plan[0] if isinstance(n_plan, tuple) else 1

    if test_type == "two_sample" and isinstance(n_plan, int):
        n_plan = (n_plan, math.ceil(ratio * n_plan))
        warnings.warn(
            f"""test_type=="two_sample" specified, but n_plan[1] not provided. n_plan[1] is set to\
             ratio = {str(ratio)} times n_plan[0] = f{str(n_plan[1])}"""
        )

    if parameter:
        delta_s = parameter
    else:
        delta_min = check_and_return_parameter(
            param_to_check=delta_min, alternative=alternative, es_min_name="delta_min"
        )
        delta_s = delta_min

    temp_result = sample_stopping_times(
        delta_true=delta_min,
        alpha=alpha,
        alternative=alternative,
        n_sim=n_sim,
        n_max=n_plan,
        ratio=ratio,
        test_type=test_type,
        parameter=delta_s,
        seed=seed,
    )

    times = temp_result["stopping_times"]

    # Note(Alexander): Break vector is 1 whenever the sample path did not stop
    # break_vector = temp_result["break_vector"]

    boot_obj_beta = compute_boot_obj(values=times, obj_type="beta", n_plan=n_plan, n_boot=n_boot)

    evalues_at_n_max = temp_result["evalues_at_n_max"]

    boot_obj_log_implied_target = compute_boot_obj(
        values=evalues_at_n_max, obj_type="log_implied_target", n_boot=n_boot
    )

    result = {
        "beta": boot_obj_beta["t0"],
        "boot_obj_beta": boot_obj_beta,
        "log_implied_target": boot_obj_log_implied_target["t0"],
        "boot_obj_log_implied_target": boot_obj_log_implied_target,
    }

    return result


def design_pilot_safe_t(
    n_plan: Union[int, Tuple[int, int]] = 50,
    alpha: float = 0.05,
    alternative: str = "two_sided",
    h0: float = 0,
    low_param: float = 0.01,
    high_param: float = 1.2,
    tol: float = 0.01,
    inverse_method: bool = True,
    paired: bool = False,
    max_iter: int = 10,
) -> Dict[str, Any]:
    check_alternative(alternative)

    n1 = n_plan if isinstance(n_plan, int) else n_plan[0]
    n2: int
    ratio = 1.0

    if isinstance(n_plan, int):
        if paired:
            warnings.warn("Paired designed specified, but n2 not provided. n2 is set to n1")
            n2 = n1
            n_plan = (n1, n2)
            test_type = "paired"
        else:
            test_type = "one_sample"

    else:
        n2 = n_plan[1]
        ratio = n2 / n1

        if paired:
            if n1 != n2:
                raise ValueError("Paired design specified, but n_plan[1] not equal n_plan[2]")

            test_type = "paired"
        else:
            test_type = "two_sample"

    result = {
        "n_plan": n_plan,
        "parameter": None,
        "es_min": None,
        "alpha": alpha,
        "beta": None,
        "alternative": alternative,
        "test_type": test_type,
        "paired": paired,
        "h0": h0,
        "sigma": None,
        "kappa": None,
        "test_type": test_type,
        "ratio": ratio,
        "low_param": None,
        "high_param": None,
        "pilot": False,
    }

    if inverse_method:

        def inv_safe_ttest_stat_alpha(x: float) -> Any:
            # TODO: try or fail with NA
            return scipy.optimize.root(
                fun=safe_ttest_stat_alpha,
                x0=1,
                t=x,
                parameter=x,
                n1=n1,
                n2=n2,
                alpha=alpha,
                alternative=alternative,
            )

        candidate_deltas = np.arange(low_param, high_param, tol)

        inv_s_to_t_thresh = np.zeros(len(candidate_deltas))

        i = 1

        while all(~inv_s_to_t_thresh) and i <= max_iter:
            inv_s_to_t_thresh = np.array(list(map(inv_safe_ttest_stat_alpha, candidate_deltas)))

            if all(~inv_s_to_t_thresh):  # TODO: this implementation is incorrect
                candidate_deltas = np.arange(
                    high_param, high_param + 2 * len(candidate_deltas) * tol, step=tol
                )
                low_param = high_param
                high_param = candidate_deltas[len(candidate_deltas)]
                i += 1

        mp_index = inv_s_to_t_thresh == min(inv_s_to_t_thresh)

        if i > 1:
            result["low_param"] = low_param
            result["high_param"] = high_param

        if mp_index == len(candidate_deltas):
            # Note(Alexander): Check that mp_index is not the last one.
            # error_msg = "The test defining delta_s is equal to high_param. Rerun with do.call \
            #     on the output object"
            low_param = high_param
            high_param = (len(candidate_deltas) - 1) * tol + low_param
            result["low_param"] = low_param
            result["high_param"] = high_param
        elif mp_index == 1:
            # error_msg = "The test defining delta_s is equal to low_param. Rerun with do.call on \
            #     the output object"
            high_param = low_param
            low_param = high_param - (len(candidate_deltas) - 1) * tol
            result["low_param"] = low_param
            result["high_param"] = high_param

        delta_s = candidate_deltas[mp_index]

        if alternative == "less":
            delta_s = -delta_s

        result["parameter"] = delta_s
        result["error"] = inv_safe_ttest_stat_alpha(candidate_deltas[mp_index])["estim.prec"]
    else:
        # TODO(Alexander): Check relation with forward method, that is, the least conservative
        # test and maximally powered
        # Perhaps trade-off? "inverse_method" refers to solving minimum of delta_s
        # \mapsto S_{delta_s}^{-1}(1/alpha)

        # TODO(Alexander): By some monotonicity can we only look at the largest or the smallest?
        #
        # designFreqT(delta_min=delta_min, alpha=alpha, beta=beta, lowN=lowN, highN=highN)
        pass

    return result


def check_and_return_n_plan(
    n_plan: Union[int, Tuple[int, int]], ratio: float = 1, test_type: str = "one_sample"
) -> Union[int, Tuple[int, int]]:
    if test_type == "two_sample" and isinstance(n_plan, int):
        n_plan = (n_plan, math.ceil(ratio * n_plan))
        warnings.warn(
            f"""test_type=="two_sample" specified, but n_plan[1] not provided. n_plan[1] = \
                ratio*n_plan[0], that is, {n_plan[1]}"""
        )
    elif test_type == "paired" and isinstance(n_plan, int):
        n_plan = (n_plan, n_plan)
        warnings.warn(
            'test_type=="paired" specified, but n_plan[1] not provided. n_plan[1] set to \
                n_plan[0].'
        )
    elif test_type == "one_sample" and not isinstance(n_plan, int):
        n_plan = n_plan[0]
        warnings.warn(
            'test_type=="one_sample" specified, but two n_plan[1] provided, which is ignored.'
        )

    return n_plan
