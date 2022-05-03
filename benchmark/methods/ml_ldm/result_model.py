# pylint: disable=invalid-name
from typing import List, Dict

from pydantic import Field
from pydantic.dataclasses import dataclass

from benchmark.task.schemas.task_scheme import AlternativeDescription, ScalesDescription, AbstractionLevelDescription, \
    ExpertDescription


@dataclass
class ResultAlternativeAssessmentDescription:
    scaleID: str = Field(...,
                         title='ID of linguistic scale')
    alternativeID: str = Field(...,
                               title='ID of alternative')
    estimation: List[Dict[str, float]] = Field(...,
                                               title='Aggregated assessment')


@dataclass
class MLLDMTaskResultDTOScheme:
    alternatives: List[AlternativeDescription]
    scales: List[ScalesDescription]
    abstractionLevels: List[AbstractionLevelDescription]
    experts: List[ExpertDescription]
    abstractionLevelWeights: Dict[str, float]
    expertWeightsRule: Dict[str, float]
    alternativesOrdered: List[ResultAlternativeAssessmentDescription]
