import torch
from typing import Tuple, Callable
from .SENSAI import sensai
import math
import profiling

def _sign(x):
    #  https://numpy.org/devdocs/reference/generated/numpy.sign.html
    return -1 if x < 0 else 0 if x == 0 else 1

def _minimize_scalar_bounded(func, x1, x2, xtol=1e-5, maxiter=500):
    # https://github.com/scipy/scipy/blob/v1.16.2/scipy/optimize/_optimize.py#L2195-L2286
    """
    Options
    -------
    maxiter : int
        Maximum number of iterations to perform.
    disp: int, optional
        If non-zero, print messages.

        ``0`` : no message printing.

        ``1`` : non-convergence notification messages only.

        ``2`` : print a message on convergence too.

        ``3`` : print iteration results.

    xtol : float
        Absolute error in solution `xopt` acceptable for convergence.

    """
    if x1 > x2:
        raise ValueError("The lower bound exceeds the upper bound.")

    sqrt_eps = math.sqrt(2.2e-16)
    golden_mean = 0.5 * (3.0 - math.sqrt(5.0))
    a, b = x1, x2
    fulc = a + golden_mean * (b - a)
    nfc, xf = fulc, fulc
    rat = e = 0.0
    x = xf
    fx = func(x)
    num = 1
    fmin_data = (1, xf, fx)
    fu = float("inf")

    ffulc = fnfc = fx
    xm = 0.5 * (a + b)
    tol1 = sqrt_eps * abs(xf) + xtol / 3.0
    tol2 = 2.0 * tol1

    while (abs(xf - xm) > (tol2 - 0.5 * (b - a))):
        golden = 1
        # Check for parabolic fit
        if abs(e) > tol1:
            golden = 0
            r = (xf - nfc) * (fx - ffulc)
            q = (xf - fulc) * (fx - fnfc)
            p = (xf - fulc) * q - (xf - nfc) * r
            q = 2.0 * (q - r)
            if q > 0.0:
                p = -p
            q = abs(q)
            r = e
            e = rat

            # Check for acceptability of parabola
            if ((abs(p) < abs(0.5*q*r)) and (p > q*(a - xf)) and
                    (p < q * (b - xf))):
                rat = (p + 0.0) / q
                x = xf + rat

                if ((x - a) < tol2) or ((b - x) < tol2):
                    si = _sign(xm - xf) + ((xm - xf) == 0)
                    rat = tol1 * si
            else:      # do a golden-section step
                golden = 1

        if golden:  # do a golden-section step
            if xf >= xm:
                e = a - xf
            else:
                e = b - xf
            rat = golden_mean*e

        si = _sign(rat) + (rat == 0)
        x = xf + si * max(abs(rat), tol1)
        fu = func(x)
        num += 1

        if fu <= fx:
            if x >= xf:
                a = xf
            else:
                b = xf
            fulc, ffulc = nfc, fnfc
            nfc, fnfc = xf, fx
            xf, fx = x, fu
        else:
            if x < xf:
                a = x
            else:
                b = x
            if (fu <= fnfc) or (nfc == xf):
                fulc, ffulc = nfc, fnfc
                nfc, fnfc = x, fu
            elif (fu <= ffulc) or (fulc == xf) or (fulc == nfc):
                fulc, ffulc = x, fu

        xm = 0.5 * (a + b)
        tol1 = sqrt_eps * abs(xf) + xtol / 3.0
        tol2 = 2.0 * tol1

        if num >= maxiter:
            break

    fval = fx
    return xf, fval

def sensai_fminbnd(
    minThreshold: float,
    maxThreshold: float,
    EEGdata_epoched: torch.Tensor,
    srate: float,
    epoch_size: float,
    refCOV: torch.Tensor,
    Eval: torch.Tensor,
    Evec: torch.Tensor,
    noise_multiplier: float,
    TolX: float = 1e-1, # tolerance for threshold optimization, default was 0.1 speed/accuracy trade-off
    skip_checks_and_return_cleaned_only: bool = False,
    maxiter: int = 500
) -> Tuple[float, float]:
    """MATLAB-style wrapper: returns (optimalThreshold, maxSENSAIScore)."""

    def objective(artifact_threshold: float) -> float:
        # minimize negative SENSAI
        profiling.mark("sensai_fminbnd_objective_call")
        try:
            _, _, score = sensai(
                EEGdata_epoched=EEGdata_epoched,
                srate=srate,
                epoch_size=epoch_size,
                artifact_threshold=float(artifact_threshold),
                refCOV=refCOV,
                Eval=Eval,
                Evec=Evec,
                noise_multiplier=noise_multiplier,
                skip_checks_and_return_cleaned_only=skip_checks_and_return_cleaned_only
            )
            return -float(score)
        except Exception as e:
            print("Objective found exception: Steering search away from threshold =", artifact_threshold)
            return float("+inf")

    xopt, fval = _minimize_scalar_bounded(
        objective, minThreshold, maxThreshold,
        xtol=float(TolX), maxiter=maxiter
    )
    profiling.mark("sensai_fminbnd_done")

    return float(xopt), float(-fval)
