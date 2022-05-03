from benchmark.task.generator.primitives.abstraction_levels import generate_abstraction_levels
from benchmark.task.generator.primitives.abstraction_levels_weight import generate_abstraction_level_weights
from benchmark.task.generator.primitives.alternatives import generate_alternatives
from benchmark.task.generator.primitives.assessments import generate_assessments
from benchmark.task.generator.primitives.criteria import generate_criteria
from benchmark.task.generator.primitives.criteria_weights import generate_criteria_weights
from benchmark.task.generator.primitives.expert_weights import generate_expert_weights
from benchmark.task.generator.primitives.experts import generate_experts
from benchmark.task.generator.primitives.scale import generate_scales
from benchmark.task.generator.task_type import TaskType
from benchmark.task.schemas.task_scheme import TaskDTOScheme


class SingleTaskGenerator:
    _num_experts: int
    _num_alternatives: int
    _num_criteria_groups: int
    _num_criteria_per_group: int
    _task_type: TaskType
    _criteria_groups: int

    def __init__(self, num_experts: int, num_alternatives: int, num_criteria_groups: int, num_criteria_per_group: int,
                 task_type: TaskType):
        self._num_experts = num_experts
        self._num_alternatives = num_alternatives
        self._num_criteria_groups = num_criteria_groups
        self._num_criteria_per_group = num_criteria_per_group
        self._task_type = task_type

    def run(self) -> TaskDTOScheme:
        criteria = generate_criteria(self._num_criteria_groups, self._num_criteria_per_group, self._task_type)
        criteria_weights = generate_criteria_weights(criteria, self._num_criteria_per_group)
        alternatives = generate_alternatives(self._num_alternatives, self._num_criteria_groups)
        scales = generate_scales(self._task_type)
        abstraction_levels = generate_abstraction_levels(self._num_criteria_groups)
        experts = generate_experts(self._num_experts)
        task_dto = TaskDTOScheme(criteria=criteria,
                                 criteriaWeightsPerGroup=criteria_weights,
                                 alternatives=alternatives,
                                 scales=scales,
                                 abstractionLevels=abstraction_levels,
                                 abstractionLevelWeights=generate_abstraction_level_weights(abstraction_levels),
                                 expertWeightsRule=generate_expert_weights(),
                                 experts=experts,
                                 estimations=generate_assessments(criteria, alternatives, experts, scales,
                                                                  self._task_type))
        return task_dto