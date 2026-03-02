"""
CoupledInterventionProtocol: two-stage cooperative geometry intervention design.

Stage 1 (Rotation)  — lifestyle intervention to rotate the patient's cooperative
                       direction w toward a treatment-receptive orientation.
Stage 2 (Treatment) — chemical/clinical treatment administered once geometric
                       alignment is verified.
"""
import numpy as np
from typing import Dict, List, Optional

from .profile import CooperativeGeometryProfile
from .distances import cooperative_direction_distance, occupation_distance


class CoupledInterventionProtocol:
    """
    Protocol for coupled interventions using cooperative geometry.

    Parameters
    ----------
    target_direction : target cooperative direction w_target (d,) from treatment responders.
    target_profile : target partition profile pi_target (K,), optional.
    rotation_threshold : max acceptable direction gap in radians for Stage 2 clearance.
                         Default pi/6 (30 degrees).
    occupation_threshold : min cooperative partition fraction for Stage 2 clearance.
                           Default 0.6 (60 % of observations in cooperative regime).
    """

    def __init__(
        self,
        target_direction: np.ndarray,
        target_profile: Optional[np.ndarray] = None,
        rotation_threshold: float = np.pi / 6,
        occupation_threshold: float = 0.6,
    ) -> None:
        self.target_direction = np.asarray(target_direction, dtype=float)
        self.target_profile = (
            np.asarray(target_profile, dtype=float)
            if target_profile is not None
            else None
        )
        self.rotation_threshold = float(rotation_threshold)
        self.occupation_threshold = float(occupation_threshold)

    # ------------------------------------------------------------------ #
    # Factory                                                              #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_responders(
        cls,
        responder_profiles: List[CooperativeGeometryProfile],
        rotation_threshold: float = np.pi / 6,
        occupation_threshold: float = 0.6,
    ) -> "CoupledInterventionProtocol":
        """
        Build a protocol from a set of treatment responders.

        The target direction is the normalised mean of responder cooperative directions
        (a Fréchet-mean approximation on the sphere).  The target profile is the
        mean occupation profile across responders who have longitudinal data.
        """
        directions = np.array([p.cooperative_direction for p in responder_profiles])
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        directions_unit = directions / np.where(norms > 0, norms, 1.0)
        w_target = directions_unit.mean(axis=0)

        profs_with_occ = [p for p in responder_profiles if p.partition_profile is not None]
        pi_target = None
        if profs_with_occ:
            pi_target = np.mean(
                [p.partition_profile for p in profs_with_occ], axis=0
            )

        return cls(
            target_direction=w_target,
            target_profile=pi_target,
            rotation_threshold=rotation_threshold,
            occupation_threshold=occupation_threshold,
        )

    # ------------------------------------------------------------------ #
    # Stage assessment                                                     #
    # ------------------------------------------------------------------ #

    def assess_readiness(self, profile: CooperativeGeometryProfile) -> Dict:
        """
        Assess whether a patient is ready for Stage 2 treatment.

        Returns a dict with keys:
          'direction_aligned', 'direction_gap', 'cooperative_fraction',
          'occupation_gap', 'recommendation'.
        """
        direction_gap = cooperative_direction_distance(
            profile.cooperative_direction, self.target_direction
        )
        direction_aligned = direction_gap <= self.rotation_threshold

        cooperative_fraction = 0.0
        occ_gap = None
        if profile.partition_profile is not None:
            K = len(profile.partition_profile)
            # Upper half of partitions by T-rank = cooperative regime.
            cooperative_fraction = float(profile.partition_profile[K // 2:].sum())
            if self.target_profile is not None:
                Ki = min(K, len(self.target_profile))
                occ_gap = occupation_distance(
                    profile.partition_profile[:Ki], self.target_profile[:Ki]
                )

        if direction_aligned and cooperative_fraction >= self.occupation_threshold:
            recommendation = "proceed_to_stage_2"
        elif direction_aligned or cooperative_fraction >= self.occupation_threshold * 0.5:
            recommendation = "continue_stage_1"
        else:
            recommendation = "stage_1_required"

        return {
            "direction_aligned": direction_aligned,
            "direction_gap": float(direction_gap),
            "cooperative_fraction": cooperative_fraction,
            "occupation_gap": occ_gap,
            "recommendation": recommendation,
        }

    def verify_rotation(
        self,
        profile_pre: CooperativeGeometryProfile,
        profile_post: CooperativeGeometryProfile,
    ) -> Dict:
        """
        Verify cooperative direction rotation after a Stage 1 intervention.

        Returns a dict with keys:
          'direction_improved', 'direction_gap_pre', 'direction_gap_post',
          'occupation_improved', 'cooperative_fraction_pre',
          'cooperative_fraction_post', 'gate_decision'.
        """
        gap_pre = cooperative_direction_distance(
            profile_pre.cooperative_direction, self.target_direction
        )
        gap_post = cooperative_direction_distance(
            profile_post.cooperative_direction, self.target_direction
        )
        direction_improved = gap_post < gap_pre

        coop_pre, coop_post, occ_improved = 0.0, 0.0, None
        if (
            profile_pre.partition_profile is not None
            and profile_post.partition_profile is not None
        ):
            K_pre = len(profile_pre.partition_profile)
            K_post = len(profile_post.partition_profile)
            coop_pre = float(profile_pre.partition_profile[K_pre // 2:].sum())
            coop_post = float(profile_post.partition_profile[K_post // 2:].sum())
            occ_improved = coop_post > coop_pre

        stage_2_ready = gap_post <= self.rotation_threshold and (
            occ_improved is None or coop_post >= self.occupation_threshold
        )

        if stage_2_ready:
            gate_decision = "proceed_to_stage_2"
        elif direction_improved:
            gate_decision = "continue_stage_1"
        else:
            gate_decision = "clinical_review"

        return {
            "direction_improved": direction_improved,
            "direction_gap_pre": float(gap_pre),
            "direction_gap_post": float(gap_post),
            "occupation_improved": occ_improved,
            "cooperative_fraction_pre": coop_pre,
            "cooperative_fraction_post": coop_post,
            "gate_decision": gate_decision,
        }
