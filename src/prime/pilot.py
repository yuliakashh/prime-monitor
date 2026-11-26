import numpy as np
from scipy import stats
from statistics import NormalDist


def n_per_group(sigma, mde, alpha=0.05, power=0.8):
    z_alpha = NormalDist().inv_cdf(1 - alpha / 2)
    z_power = NormalDist().inv_cdf(power)

    n = 2 * ((z_alpha + z_power) * sigma / mde) ** 2

    return int(np.ceil(n))


def mde_for_n(sigma, n, alpha=0.05, power=0.8):
    z_alpha = NormalDist().inv_cdf(1 - alpha / 2)
    z_power = NormalDist().inv_cdf(power)

    mde = (z_alpha + z_power) * sigma * np.sqrt(2 / n)

    return float(mde)


def diff_ci(t, c, alpha=0.05):
    d = t.mean() - c.mean()

    se = (
        t.var(ddof=1) / len(t)
        + c.var(ddof=1) / len(c)
    ) ** 0.5

    q = NormalDist().inv_cdf(1 - alpha / 2)

    lo = d - q * se
    hi = d + q * se

    p = stats.ttest_ind(t, c, equal_var=False).pvalue

    return {
        "diff": float(d),
        "lo": float(lo),
        "hi": float(hi),
        "p": float(p),
    }


def srm_pvalue(sizes):
    sizes = np.asarray(sizes)

    return float(
        stats.chisquare(sizes).pvalue
    )


def cuped(y, x):
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)

    theta = np.cov(y, x, ddof=1)[0, 1] / np.var(x, ddof=1)

    y_cuped = y - theta * (x - x.mean())

    return y_cuped, float(theta)