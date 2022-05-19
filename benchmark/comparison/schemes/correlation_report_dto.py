from typing import Optional

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
    num_replicas: Optional[int]
    generate_new_dataset: Optional[bool] = True
    execute_decision_makers: Optional[bool] = True
    calculate_correlation_reports: Optional[bool] = True
    equal_expert_weights: Optional[bool] = True
    generate_concrete_expert_weights: Optional[bool] = True


@dataclass
class CorrelationReportDTO:
    unique_configurations: list[UniqueExperimentalSetupDTO]
    experiment_info: ExperimentInfoDTO
    fails: Optional[dict[str, int]] = None

    class Config:
        arbitrary_types_allowed = True
