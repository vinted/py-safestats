import numpy as np
import pytest
import scipy
from rpy2 import robjects as ro

from src.ttest import (
    check_and_return_parameter,
    compute_boot_obj,
    compute_confint_t,
    compute_n_plan,
    compute_n_plan_batch,
    define_ttest_n,
    design_safe_t,
    generate_normal_data,
    safe_ttest,
    safe_ttest_stat,
    sample_stopping_times,
)


def r_obj_to_dict(obj):
    try:
        d = dict(zip(obj.names, list(obj)))
    except TypeError:
        if len(obj) == 1:
            return obj[0]
        else:
            raise ValueError("Unknown object; needs intervention")
    for i, k in enumerate(d.keys()):
        try:
            if len(d[k]):
                try:
                    d[k] = np.asarray(d[k], dtype=object)
                except ValueError:
                    pass
                if len(d[k]) == 1:
                    d[k] = d[k][0]
        except TypeError:
            d[k] = None

    return d


def check_r_python_equal(r_obj, py_obj, diff=1e-9):
    if isinstance(r_obj, (float, int, np.int64)):
        if np.abs(r_obj - py_obj) < diff:
            return True
    elif isinstance(r_obj, (list, np.ndarray, tuple)):
        return scipy.stats.ks_2samp(r_obj, py_obj).statistic == diff
    else:
        assert r_obj.keys() == py_obj.keys()
        for k in r_obj.keys():
            if r_obj[k] is not None:
                if isinstance(r_obj[k], (int, np.int64, float)):
                    assert r_obj[k] == py_obj[k]
                else:  # if numpy array
                    assert (r_obj[k] == py_obj[k]).all()

        return True


@pytest.mark.rtest
def test_genhypergeo():
    n1, delta_s, t = 50, 0.2, 2
    n_eff, nu = n1, n1 - 1
    a = t**2 / (nu + t**2)
    z_arg = (-1 / 2) * a * n_eff * delta_s**2

    r_object = r_obj_to_dict(
        ro.r(
            f"""
        aKummerFunction = Re(hypergeo::genhypergeo(U=-{nu}/2, L=1/2, {z_arg}))
    """
        )
    )

    a_kummer = scipy.special.hyp1f1(-nu / 2, 1 / 2, z_arg)
    assert check_r_python_equal(r_object, a_kummer)


@pytest.mark.rtest
def test_define_ttest_n():
    low_n, high_n, ratio, test_type_r, test_type_py = 3, 100, 1, "'paired'", "paired"

    r_object = r_obj_to_dict(
        ro.r(
            f"""
        low_n <- {low_n}
        high_n <- {high_n}
        ratio <- {ratio}
        test_type <- {test_type_r}
        safestats:::defineTTestN("lowN"=low_n, "highN"=high_n, "ratio"=ratio, "testType"=test_type)
    """
        )
    )
    py_object = define_ttest_n(low_n=low_n, high_n=high_n, ratio=ratio, test_type=test_type_py)
    r_object["n_eff"] = r_object.pop("nEff")

    assert check_r_python_equal(r_object, py_object)


@pytest.mark.rtest
def test_safe_ttest_stat():
    t, parameter, n1, n2 = -3, 0.25, 300, 300
    alternative_r, alternative_py = "'greater'", "greater"

    r_result = r_obj_to_dict(
        ro.r(
            f"""
        t <- {t}
        parameter <- {parameter}
        n1 <- {n1}
        n2 <- {n2}
        alternative <- {alternative_r}
        safestats:::safeTTestStat("t"=t, "parameter"=parameter, "n1"=n1, "n2"=n2,
        "alternative"=alternative)
    """
        )
    )

    py_result = safe_ttest_stat(t=t, parameter=parameter, n1=n1, n2=n2, alternative=alternative_py)

    assert check_r_python_equal(r_result, py_result)


@pytest.mark.rtest
def test_compute_n_plan_batch():
    delta_min, low_n, high_n, ratio, alpha, beta = 0.2, 3, 1000, 1, 0.05, 0.2
    alternative_r, alternative_py = "'greater'", "greater"
    test_type_r, test_type_py = "'paired'", "paired"

    r_result = r_obj_to_dict(
        ro.r(
            f"""
        delta_min <- {delta_min}
        low_n <- {low_n}
        high_n <- {high_n}
        ratio <- {ratio}
        alternative <- {alternative_r}
        alpha <- {alpha}
        beta <- {beta}
        test_type <- {test_type_r}
        safestats:::computeNPlanBatchSafeT("deltaMin"=delta_min, "alpha"=alpha, "beta"=beta,
                                            "alternative"=alternative, "testType"=test_type,
                                            "lowN"=low_n, "highN"=high_n, "ratio"=ratio)
    """
        )
    )

    py_result = compute_n_plan_batch(
        delta_min=delta_min,
        low_n=low_n,
        high_n=high_n,
        ratio=ratio,
        alternative=alternative_py,
        alpha=alpha,
        beta=beta,
        test_type=test_type_py,
    )
    r_result["n_plan"] = r_result.pop("nPlan")
    r_result["delta_s"] = r_result.pop("deltaS")

    assert check_r_python_equal(r_result, py_result)


@pytest.mark.rtest
def test_generate_normal_data():
    n_plan, n_sim = 300, 1000
    delta_true, mu_global, sigma_true = 0.5, 0, 1
    paired_r, paired_py = "FALSE", False
    seed = 123456
    mu_true = "NULL"

    r_result = r_obj_to_dict(
        ro.r(
            f"""
        n_plan <- {n_plan}
        delta_true <- {delta_true}
        n_sim <- {n_sim}
        mu_global <- {mu_global}
        sigma_true <- {sigma_true}
        paired <- {paired_r}
        seed <- {seed}
        mu_true <- {mu_true}
        safestats:::generateNormalData("nPlan"=n_plan, "nSim"=n_sim, "deltaTrue"=delta_true,
                                        "muGlobal"=mu_global, "sigmaTrue"=sigma_true,
                                        "paired"=paired, "seed"=seed, "muTrue"=mu_true)
    """
        )
    )

    py_result = generate_normal_data(
        n_plan=n_plan,
        n_sim=n_sim,
        delta_true=delta_true,
        mu_global=mu_global,
        sigma_true=sigma_true,
        paired=paired_py,
        seed=seed,
    )

    n_r = r_result["dataGroup1"].mean(axis=1)
    x_py = py_result["data_group1"].mean(axis=1)

    assert check_r_python_equal(n_r, x_py, 0.03)


@pytest.mark.rtest
def test_check_and_return_parameter():
    delta_true = 0.2
    alternative_r, alternative_py = "'greater'", "greater"
    es_min_name_r, es_min_name_py = "'deltaTrue'", "delta_true"

    r_result = r_obj_to_dict(
        ro.r(
            f"""
        parameter <- {delta_true}
        alternative <- {alternative_r}
        esMinName <- {es_min_name_r}
        safestats:::checkAndReturnsEsMinParameterSide("paramToCheck"=parameter,
                            "esMinName"=esMinName, "alternative"=alternative, paramDomain=NULL)
    """
        )
    )

    py_result = check_and_return_parameter(
        param_to_check=delta_true,
        alternative=alternative_py,
        es_min_name=es_min_name_py,
        param_domain=None,
    )

    assert check_r_python_equal(r_result, py_result)


@pytest.mark.rtest
def test_sample_stopping_times():
    delta_min, ratio = 0.2, 1
    alpha, beta, seed, n_sim, n_plan_batch = 0.05, 0.2, 123456, 10, (68, 68)
    alternative_r, alternative_py = "'greater'", "greater"
    delta_s, test_type_r, test_type_py = delta_min, "'paired'", "paired"

    r_result = r_obj_to_dict(
        ro.r(
            f"""
        delta_min <- {delta_min}
        ratio <- {ratio}
        alternative <- {alternative_r}
        alpha <- {alpha}
        beta <- {beta}
        test_type <- {test_type_r}
        delta_s <- {delta_s}
        seed <- {seed}
        n_sim <- {n_sim}

        n_plan_batch <- c{n_plan_batch}

        samplingResults <- safestats:::sampleStoppingTimesSafeT("deltaTrue"=delta_min,
                                        "alpha"=alpha, "alternative" = alternative, "seed"=seed,
                                        "nSim"=n_sim, "nMax"=n_plan_batch, "ratio"=ratio,
                                        "testType"=test_type, "parameter"=delta_s)
    """
        )
    )

    py_result = sample_stopping_times(
        delta_true=delta_min,
        alpha=alpha,
        alternative=alternative_py,
        seed=seed,
        n_sim=n_sim,
        n_max=n_plan_batch,
        ratio=ratio,
        test_type=test_type_py,
        parameter=delta_s,
    )

    n_r, x_py = r_result["stoppingTimes"], py_result["stopping_times"]

    assert check_r_python_equal(n_r, x_py, 0.2)


@pytest.mark.rtest
def test_compute_boot_obj():
    beta, seed, n_boot = 0.2, 123456, 10

    times = (
        101,
        65,
        120,
        143,
        297,
        214,
        56,
        89,
        195,
        295,
        102,
        91,
        138,
        101,
        158,
        153,
        105,
        61,
        49,
        140,
        229,
        39,
        44,
        106,
        39,
        194,
        60,
        35,
        297,
        83,
        163,
        66,
        145,
        64,
        297,
        75,
        25,
        195,
        94,
        297,
        77,
        195,
        155,
        121,
        57,
        198,
        67,
        297,
        84,
        296,
        43,
        87,
        73,
        48,
        252,
        84,
        114,
        51,
        146,
        153,
        230,
        297,
        203,
        58,
        134,
        82,
        237,
        75,
        181,
        186,
        68,
        148,
        65,
        61,
        37,
        137,
        127,
        134,
        44,
        297,
        58,
        96,
        118,
        49,
        105,
        112,
        171,
        145,
        116,
        138,
        59,
        75,
        93,
        38,
        71,
        46,
        37,
        297,
        160,
        297,
        234,
        207,
        60,
        73,
        178,
        132,
        31,
        94,
        69,
        67,
        97,
        297,
        141,
        37,
        156,
        58,
        297,
        188,
        134,
        200,
        248,
        101,
        33,
        88,
        116,
        243,
        274,
        191,
        112,
        144,
        47,
        168,
        199,
        297,
        204,
        179,
        260,
        121,
        39,
        41,
        255,
        198,
        149,
        172,
        293,
        261,
        217,
        111,
        135,
        54,
        297,
        250,
        82,
        297,
        123,
        127,
        125,
        141,
        206,
        38,
        70,
        77,
        85,
        93,
        219,
        183,
        212,
        84,
        124,
        170,
        55,
        87,
        297,
        195,
        297,
        72,
        169,
        187,
        52,
        223,
        64,
        145,
        80,
        201,
        74,
        297,
        37,
        255,
        189,
        139,
        126,
        297,
        227,
        297,
        177,
        296,
        110,
        179,
        97,
        110,
        67,
        150,
        55,
        67,
        297,
        88,
        59,
        63,
        288,
        297,
        68,
        284,
        230,
        104,
        163,
        266,
        32,
        297,
        195,
        137,
        142,
        28,
        79,
        217,
        36,
        73,
        73,
        267,
        210,
        297,
        112,
        107,
        36,
        49,
        114,
        111,
        236,
        297,
        53,
        60,
        119,
        295,
        92,
        59,
        70,
        297,
        174,
        61,
        189,
        145,
        297,
        243,
        159,
        105,
        129,
        102,
        88,
        297,
        296,
        80,
        297,
        71,
        221,
        273,
        297,
        50,
        75,
        297,
        254,
        191,
        79,
        83,
        297,
        76,
        297,
        78,
        190,
        297,
        102,
        147,
        52,
        80,
        297,
        150,
        42,
        30,
        57,
        177,
        40,
        104,
        177,
        51,
        127,
        52,
        129,
        215,
        152,
        189,
        127,
        221,
        206,
        297,
        264,
        60,
        79,
        297,
        215,
        158,
        84,
        57,
        169,
        66,
        46,
        65,
        116,
        101,
        151,
        144,
        98,
        117,
        297,
        86,
        39,
        54,
        238,
        297,
        280,
        148,
        297,
        199,
        297,
        62,
        93,
        278,
        194,
        253,
        131,
        128,
        85,
        297,
        166,
        209,
        297,
        49,
        96,
        83,
        181,
        191,
        156,
        70,
        87,
        75,
        144,
        74,
        297,
        156,
        126,
        106,
        172,
        94,
        131,
        134,
        128,
        111,
        297,
        119,
        58,
        161,
        110,
        297,
        49,
        34,
        75,
        129,
        286,
        64,
        204,
        132,
        88,
        161,
        145,
        54,
        112,
        238,
        297,
        49,
        225,
        287,
        185,
        217,
        297,
        297,
        62,
        140,
        189,
        272,
        118,
        106,
        66,
        113,
        120,
        76,
        296,
        45,
        49,
        297,
        297,
        224,
        87,
        39,
        87,
        61,
        75,
        130,
        143,
        64,
        59,
        135,
        297,
        76,
        205,
        41,
        297,
        49,
        105,
        132,
        149,
        89,
        65,
        297,
        297,
        57,
        105,
        250,
        58,
        60,
        297,
        99,
        61,
        132,
        96,
        41,
        297,
        197,
        48,
        58,
        124,
        90,
        30,
        297,
        288,
        67,
        60,
        50,
        143,
        48,
        113,
        37,
        109,
        297,
        297,
        59,
        297,
        136,
        107,
        100,
        73,
        50,
        55,
        77,
        99,
        141,
        297,
        138,
        110,
        122,
        98,
        58,
        58,
        184,
        148,
        244,
        125,
        211,
        63,
        152,
        143,
        137,
        192,
        250,
        59,
        93,
        241,
        231,
        112,
        297,
        154,
        33,
        37,
        155,
        297,
        68,
        84,
        277,
        208,
        71,
        257,
        67,
        63,
        297,
        297,
        162,
        62,
        119,
        153,
        31,
        297,
        123,
        149,
        297,
        122,
        51,
        230,
    )

    r_result = r_obj_to_dict(
        ro.r(
            f"""
        set.seed({seed})

        n_boot <- {n_boot}
        beta <- {beta}
        times <- c{times}

        bootObjN1Plan <- safestats:::computeBootObj("values"=times, "objType"="nPlan",
                                                    "beta"=beta, "nBoot"=n_boot)
    """
        )
    )

    times = np.array(times)
    py_result = compute_boot_obj(
        values=times, obj_type="n_plan", beta=beta, n_boot=n_boot, seed=seed
    )

    n_r, x_py = r_result["t"].reshape(-1), np.array(py_result["t"])

    assert check_r_python_equal(n_r, x_py, 0.2)


@pytest.mark.rtest
def test_compute_n_plan():
    alpha = 0.05
    beta = 0.2
    delta_min = 9 / (np.sqrt(2) * 15)
    n_sim = 10
    n_boot = 10
    low_n = 3
    high_n = 1000
    n_seeds = 50

    r_nplanbatch, r_n_plan, r_nmean = np.empty(n_seeds), np.empty(n_seeds), np.empty(n_seeds)
    py_nplanbatch, py_n_plan, py_nmean = np.empty(n_seeds), np.empty(n_seeds), np.empty(n_seeds)

    for seed in range(n_seeds):
        r_result = r_obj_to_dict(
            ro.r(
                f"""
            tempResult <- safestats:::computeNPlanSafeT("deltaMin"={delta_min}, "beta"={beta},
                                "alpha"={alpha}, "alternative"='greater', "testType"='paired',
                                "lowN"={low_n}, "highN"={high_n}, "ratio"=1, "seed"={seed},
                                "nSim"={n_sim}, "nBoot"={n_boot}, "pb"=FALSE)
        """
            )
        )
        r_n_plan[seed] = r_result["n1Plan"]
        r_nmean[seed] = r_result["n1Mean"]
        r_nplanbatch[seed] = r_result["nPlanBatch"][0]

        py_result = compute_n_plan(
            delta_min=delta_min,
            beta=beta,
            alpha=alpha,
            alternative="greater",
            test_type="paired",
            low_n=low_n,
            high_n=high_n,
            ratio=1,
            seed=seed,
            n_sim=n_sim,
            n_boot=n_boot,
        )

        py_n_plan[seed] = py_result["n1_plan"]
        py_nmean[seed] = py_result["n1_mean"]
        py_nplanbatch[seed] = py_result["n_plan_batch"][0]

    assert check_r_python_equal(r_n_plan, py_n_plan, 0.18)
    assert check_r_python_equal(r_nmean, py_nmean, 0.18)
    assert check_r_python_equal(r_nplanbatch, py_nplanbatch, 0.0)


@pytest.mark.rtest
def test_design_safe_t():
    n_sim = 100
    n_boot = 100
    delta_min = 0.2
    alternative_r = "'twoSided'"
    alternative_py = "two_sided"
    alpha = 0.05
    beta = 0.2
    test_type_r = "'twoSample'"
    test_type_py = "two_sample"

    r_n_plan, py_n_plan = list(), list()

    for seed in range(0, 10):
        r_result = r_obj_to_dict(
            ro.r(
                f"""
        design_obj <- safestats:::designSafeT(deltaMin={delta_min}, alpha={alpha}, beta={beta},
                            alternative={alternative_r}, testType={test_type_r},
                            nSim={n_sim}, n_boot={n_boot}, seed={seed}, pb=FALSE)
        """
            )
        )
        r_n_plan.append(r_result["nPlan"][0])

        py_result = design_safe_t(
            delta_min=delta_min,
            alpha=alpha,
            beta=beta,
            alternative=alternative_py,
            test_type=test_type_py,
            n_sim=n_sim,
            n_boot=n_boot,
            seed=seed,
        )
        py_n_plan.append(py_result["n_plan"][0])

    assert check_r_python_equal(r_n_plan, py_n_plan, 0.4)


@pytest.mark.rtest
def test_safe_ttest():
    alpha = 0.05
    beta = 0.2
    delta_min = 9 / (np.sqrt(2) * 15)
    seed = 123456

    r_statistic, py_statistic = list(), list()
    r_evalue, py_evalue = list(), list()

    design_obj = design_safe_t(
        delta_min=delta_min,
        alpha=alpha,
        beta=beta,
        alternative="greater",
        test_type="paired",
        seed=seed,
    )

    ro.r(
        f"""
        designObj <- safestats:::designSafeT(deltaMin={delta_min}, alpha={alpha}, beta={beta},
                                alternative="greater", testType="paired", seed={seed}, pb=FALSE)
    """
    )

    for _ in range(0, 10):
        pre_data = np.random.normal(loc=120, scale=15, size=design_obj["n_plan"][0])
        post_data = np.random.normal(loc=111, scale=15, size=design_obj["n_plan"][1])

        py_result = safe_ttest(
            x=pre_data, y=post_data, alternative="greater", design_obj=design_obj, paired=True
        )

        r_result = r_obj_to_dict(
            ro.r(
                f"""

            safeT <- safestats:::safeTTest(x={'c'+str(tuple(pre_data))},
                                           y={'c'+str(tuple(post_data))},
                                           alternative = "greater", designObj=designObj,
                                           paired=TRUE, pb=FALSE)
        """
            )
        )

        r_statistic.append(r_result["statistic"])
        py_statistic.append(py_result["statistic"])
        r_evalue.append(r_result["eValue"])
        py_evalue.append(py_result["e_value"])

    assert check_r_python_equal(r_statistic, py_statistic, 0.1)


@pytest.mark.rtest
def test_compute_confint_t():
    mean_obs, sd_obs, n_eff, nu, delta_s = 120, 15, 100, 99, 3
    py_result = compute_confint_t(mean_obs, sd_obs, n_eff, nu, delta_s)
    r_result = ro.r(
        f"""
        confInf <- safestats:::computeConfidenceIntervalT({mean_obs}, {sd_obs}, {n_eff}, {nu},
                                                            {delta_s})
    """
    )
    r_result = (r_result[0], r_result[1])

    assert check_r_python_equal(py_result, r_result, 0)
