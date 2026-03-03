"""
AutoITE — Individual Treatment Effect estimation via Intrinsic Causal Geometry.

Two geometry variants share the same API:

  ICG-HVRT  geometry='cone'    SD whitening (Sigma^{-1/2}); best for clean Gaussian data.
  ICG-HART  geometry='pyramid' MAD whitening (1.4826*MAD); robust to outlier spikes.

Quick start
-----------
    from autoite import ICGHVRTEstimator

    est = ICGHVRTEstimator(geometry='cone', k=10).fit(X_list, T_list, Y_list)
    tau = est.predict_effect(X_new, T_new)

Full API
--------
    from autoite import (
        ICGHVRTEstimator,
        ICGHVRTMatcher,
        ConeIdentity,
        CooperativeGeometryProfile,
        SharedHVRT,
        fit_shared_hvrt,
        pool_whitened_observations,
        CoupledInterventionProtocol,
    )
"""

from .distances import ConeIdentity
from .profile import CooperativeGeometryProfile, SharedHVRT, fit_shared_hvrt, pool_whitened_observations
from .matcher import ICGHVRTMatcher
from .estimator import ICGHVRTEstimator
from .protocol import CoupledInterventionProtocol

__version__ = "2.1.0"

__all__ = [
    "ConeIdentity",
    "CooperativeGeometryProfile",
    "SharedHVRT",
    "fit_shared_hvrt",
    "pool_whitened_observations",
    "ICGHVRTMatcher",
    "ICGHVRTEstimator",
    "CoupledInterventionProtocol",
]
