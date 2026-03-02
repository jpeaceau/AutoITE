"""
AutoITE — Automatic Individual Treatment Effect Estimation
via Intrinsic Causal Geometry on Personal Quadratic Manifolds (ICG-HVRT).

Quick start
-----------
    from autoite import (
        CooperativeGeometryProfile,
        SharedHVRT,
        fit_shared_hvrt,
        ICGHVRTMatcher,
        ICGHVRTEstimator,
        CoupledInterventionProtocol,
    )
"""

from .profile import CooperativeGeometryProfile, SharedHVRT, fit_shared_hvrt
from .matcher import ICGHVRTMatcher
from .estimator import ICGHVRTEstimator
from .protocol import CoupledInterventionProtocol

__version__ = "2.0.0"

__all__ = [
    "CooperativeGeometryProfile",
    "SharedHVRT",
    "fit_shared_hvrt",
    "ICGHVRTMatcher",
    "ICGHVRTEstimator",
    "CoupledInterventionProtocol",
]
