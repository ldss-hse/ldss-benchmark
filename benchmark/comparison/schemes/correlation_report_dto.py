from pydantic.dataclasses import dataclass

from benchmark.comparison.schemes.unique_experimental_setup import UniqueExperimentalSetupDTO
from benchmark.task.generator.task_type import TaskType


@dataclass
class ExperimentInfoDTO:
    task_type: TaskType
    alternatives_range: list[int]
    criteria_range: list[int]
    num_experts: int
    num_criteria_groups: int


@dataclass
class CorrelationReportDTO:
    unique_configurations: list[UniqueExperimentalSetupDTO]
    experiment_info: ExperimentInfoDTO

    class Config:
        arbitrary_types_allowed = True