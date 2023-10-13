from typing import Any, Dict, Tuple, Union

import numpy as np
import numpy.typing as npt
from scipy import stats


def calculate_sequential_e(
    a_sample: Tuple[npt.NDArray[np.int64]],
    b_sample: Tuple[npt.NDArray[np.int64]] = None,
    prior_values: Dict[str, Any] = None,
    na_sample: np.array = None,
    nb_sample: np.array = None,
    sim_setting: bool = False,
    implied_target_setting: bool = False,
    alpha_sim: float = 0.05,
    test_type: str = "one-sided",
) -> Dict[str, Any]:
    # Check the test type
    if test_type not in ["one-sided", "two-sided"]:
        raise ValueError("Invalid value for test_type")

    # Check if na_sample and nb_sample are provided
    if na_sample is None or (test_type == "two-sided" and nb_sample is None):
        raise ValueError("Provide na_sample and nb_sample for variable group sizes")

    # Unpack the prior values
    beta_a1 = prior_values["beta_a1"]
    beta_a2 = prior_values["beta_a2"]

    # Set starting E variable
    if test_type == "one-sided":
        e_variable = update_e_proportions(
            total_success_a=0,
            total_fail_a=0,
            na=na_sample[0],  # Initialize with the first group size
            beta_a1=beta_a1,
            beta_a2=beta_a2,
        )

        total_success_a = np.cumsum(a_sample)
        group_size_vec_a = np.cumsum(na_sample)
        total_fail_a = group_size_vec_a - total_success_a

    # Additional code for one-sided test if needed
    elif test_type == "two-sided":
        beta_b1 = prior_values["beta_b1"]
        beta_b2 = prior_values["beta_b2"]

        e_variable = update_e_proportions(
            total_success_a=0,
            total_fail_a=0,
            total_success_b=0,
            total_fail_b=0,
            na=na_sample[0],  # Initialize with the first group size for a
            nb=nb_sample[0],  # Initialize with the first group size for b
            beta_a1=beta_a1,
            beta_a2=beta_a1,
            beta_b1=beta_b1,
            beta_b2=beta_b2,
        )
        # Additional code for two-sided test if needed

        total_success_a = np.cumsum(a_sample)
        total_success_b = np.cumsum(b_sample)
        group_size_vec_a = np.cumsum(na_sample)
        group_size_vec_b = np.cumsum(nb_sample)
        total_fail_a = group_size_vec_a - total_success_a
        total_fail_b = group_size_vec_b - total_success_b

    # Initialize current E variable
    current_e = 1
    stop_time = None
    stop_e = None
    e_values = np.zeros(shape=len(a_sample))

    # Iterate over the samples
    for i in range(len(a_sample)):
        # One-sided test logic
        if test_type == "one-sided":
            new_e = calculate_e_proportions(
                na1=a_sample[i],
                na=na_sample[i],
                theta_a=e_variable["theta_a"],
                theta0=e_variable["theta0"],
            )
        # Two-sided test logic
        elif test_type == "two-sided":
            new_e = calculate_e_proportions(
                na1=a_sample[i],
                na=na_sample[i],
                nb1=b_sample[i],
                nb=nb_sample[i],
                theta_a=e_variable["theta_a"],
                theta_b=e_variable["theta_b"],
                theta0=e_variable["theta0"],
            )

        current_e = new_e * current_e

        if test_type == "one-sided":
            e_variable = update_e_proportions(
                total_success_a=total_success_a[i],
                total_fail_a=total_fail_a[i],
                na=na_sample[i],
                beta_a1=beta_a1,
                beta_a2=beta_a2,
            )
        elif test_type == "two-sided":
            e_variable = update_e_proportions(
                total_success_a=total_success_a[i],
                total_fail_a=total_fail_a[i],
                total_success_b=total_success_b[i],
                total_fail_b=total_fail_b[i],
                na=na_sample[i],
                nb=nb_sample[i],
                beta_a1=beta_a1,
                beta_a2=beta_a2,
                beta_b1=beta_b1,
                beta_b2=beta_b2,
            )

        e_values[i] = current_e

        # In simulation setting, only interested in the stopping time
        if sim_setting and current_e >= (1 / alpha_sim) and stop_time is None:
            if implied_target_setting:
                # We save the time and E-value where we would have stopped, but continue collecting
                # until we reach n_plan for calculating impliedTarget
                stop_time = i
                stop_e = current_e
            else:
                stop_time = i
                stop_e = current_e
                break

    # Final processing
    if not sim_setting:
        return e_values
    else:
        if stop_time is None:
            stop_time = len(a_sample)
        if stop_e is None:
            stop_e = current_e
        return {"stop_time": stop_time, "stop_e": stop_e, "final_e": current_e}


def calculate_e_proportions(
    na1: int,
    na: int,
    nb1: Union[int, Any] = None,
    nb: Union[int, Any] = None,
    theta_a: Union[float, Any] = None,
    theta_b=None,
    theta0=None,
    test_type="one-sided",
) -> float:
    if test_type == "one-sided":
        if theta_a is None or theta0 is None:
            raise ValueError("Please provide theta_a and theta0 for one-sided test.")
        return np.exp(
            na1 * np.log(theta_a)
            + (na - na1) * np.log(1 - theta_a)
            - (na1) * np.log(theta0)
            - (na - na1) * np.log(1 - theta0)
        )
    elif test_type == "two-sided":
        if theta_a is None or theta_b is None or theta0 is None:
            raise ValueError("Please provide theta_a, theta_b, and theta0 for two-sided test.")
        return np.exp(
            na1 * np.log(theta_a)
            + (na - na1) * np.log(1 - theta_a)
            + nb1 * np.log(theta_b)
            + (nb - nb1) * np.log(1 - theta_b)
            - (na1 + nb1) * np.log(theta0)
            - (na + nb - na1 - nb1) * np.log(1 - theta0)
        )
    else:
        raise ValueError("Invalid test type. Must be either 'one-sided' or 'two-sided'.")


def update_e_proportions(
    total_success_a: int,
    total_fail_a: int,
    beta_a1: Union[float, int],
    beta_a2: Union[float, int],
    total_success_b: Union[int, Any] = None,
    total_fail_b: Union[int, Any] = None,
    na: Union[int, Any] = None,
    nb: Union[int, Any] = None,
    beta_b1: Union[int, float, Any] = None,
    beta_b2: Union[int, float, Any] = None,
    test_type: str = "one-sided",
    theta0: float = 1 / 2,
) -> Dict[str, Any]:
    theta_a = bernoulli_ml_proportion(
        total_success=total_success_a,
        total_fail=total_fail_a,
        prior_success=beta_a1,
        prior_fail=beta_a2,
    )

    if test_type == "one-sided":

        return {"theta_a": theta_a, "theta0": theta0}

    elif test_type == "two-sided":
        if (
            total_success_b is None
            or total_fail_b is None
            or na is None
            or nb is None
            or beta_b1 is None
            or beta_b2 is None
        ):
            raise ValueError("Please provide all required parameters for a two-sided test.")

        theta_b = bernoulli_ml_proportion(
            total_success=total_success_b,
            total_fail=total_fail_b,
            prior_success=beta_b1,
            prior_fail=beta_b2,
        )

        theta0 = (na * theta_a + nb * theta_b) / (na + nb)

        return {"theta_a": theta_a, "theta_b": theta_b, "theta0": theta0}

    else:
        raise ValueError("Invalid test type. Must be either 'one-sided' or 'two-sided'.")


def bernoulli_ml_proportion(
    total_success: int, total_fail: int, prior_success: np.int64 = 1, prior_fail: np.int64 = 1
):
    theta = (total_success + prior_success) / (
        total_success + total_fail + prior_success + prior_fail
    )
    return theta


def simulate_1x2_stopping(
    p, size, n_sims, na, prior_values, sim_setting, alpha
):
    results = list()

    for i in range(n_sims):
        x1 = stats.bernoulli.rvs(p=p, size=size)

        e_val = calculate_sequential_e(
            x1,
            prior_values=prior_values,
            na=na,
            sim_setting=sim_setting,
            alpha_sim=alpha,
        )

        results.append(e_val)

    safe_stopping_times = np.array([result["stop_time"] for result in results])

    return safe_stopping_times


def simulate_2x2_stopping(safe_design, M, theta_a, theta_b):
    stopping_times = np.zeros(M)
    stop_es = np.zeros(M)

    for i in range(M):
        ya = np.random.binomial(
            size=int(safe_design["n_plan"]["n_blocks_plan"]),
            p=theta_a,
            n=safe_design["n_plan"]["na"],
        )
        yb = np.random.binomial(
            size=int(safe_design["n_plan"]["n_blocks_plan"]),
            p=theta_b,
            n=safe_design["n_plan"]["nb"],
        )

        sim_result = calculate_sequential_e(
            a_sample=ya,
            b_sample=yb,
            prior_values=safe_design["beta_prior_parameter_values"],
            restriction=safe_design["restriction"],
            delta=safe_design["delta"],
            na=safe_design["n_plan"]["na"],
            nb=safe_design["n_plan"]["nb"],
            sim_setting=True,
            alpha_sim=safe_design["alpha"],
        )

        stopping_times[i] = sim_result["stop_time"]
        stop_es[i] = sim_result["stop_e"]

    all_safe_decisions = stop_es >= (1 / safe_design["alpha"])

    safe_sim = {
        "power_optio_stop": np.mean(all_safe_decisions),
        "n_mean": np.mean(stopping_times),
        "prob_less_n_design": np.mean(stopping_times < safe_design["n_plan"]["n_blocks_plan"]),
        "low_n": np.min(stopping_times),
        "e_values": stop_es,
    }

    safe_sim["all_n"] = stopping_times
    safe_sim["all_safe_decisions"] = all_safe_decisions
    safe_sim["all_rejected_n"] = stopping_times[all_safe_decisions]

    return safe_sim
