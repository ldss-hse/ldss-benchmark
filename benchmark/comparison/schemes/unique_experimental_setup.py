from typing import Optional

from pydantic.dataclasses import dataclass


def _create_from_json(other):
    return SetupInfoDTO(
        num_experts=other['num_experts'],
        num_alternatives=other['num_alternatives'],
        num_criteria=other['num_criteria'],
        num_criteria_groups=other['num_criteria_groups']
    )


@dataclass
class SetupInfoDTO:
    num_experts: int
    num_alternatives: int
    num_criteria: int
    num_criteria_groups: int

    def __eq__(self, other):
        return (
                other['num_experts'] == self.num_experts and
                other['num_alternatives'] == self.num_alternatives and
                other['num_criteria'] == self.num_criteria and
                other['num_criteria_groups'] == self.num_criteria_groups
        )


@dataclass
class UniqueExperimentalSetupDTO:
    setup_info: SetupInfoDTO
    kendall_coefficients: Optional[dict[str, float]]
    spearman_coefficients: Optional[dict[str, float]]
    top_1_matches: Optional[dict[str, int]]
    total_runs: Optional[int] = None

    class Config:
        arbitrary_types_allowed = True
