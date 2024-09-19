"""
Prototype domain models for a multi-sector projection system.
"""

from dataclasses import dataclass
from typing import Any

from muuttaa import TransformationStrategy, Projector, SegmentWeights

# TODO: Once tested, these are candidates to move to muuttaa.


@dataclass(frozen=True)
class Sector:
    id: str
    transformations: set(TransformationStrategy)
    impact_model: Projector
    valuation_model: Projector


# TODO: Does the name "Projector" make sense in this context?
# TODO: Use abstract and note concrete objects in muuttaa typing and signatures.


@dataclass(frozen=True)
class ProjectionSystem:
    common_data: Any  # Idea is good to share same inputs across sectors but not sure on implementation. Maybe have as input parameter?
    segment_weights: SegmentWeights
    transformer: (
        Any  # TODO: Need this signature properly named or protocol'd in muuttaa!
    )
    impact_projector: (
        Any  # TODO: Need this signature properly named or protocol'd in muuttaa!
    )
    value_projector: (
        Any  # TODO: Need this signature properly named or protocol'd in muuttaa!
    )


# Playground to feel this out.
projection_system.transform_data(mortality).project_impacts(parameters).value_impacts(
    parameters
)
