import random

from benchmark.task.generator.task_type import TaskType
from benchmark.task.schemas.task_scheme import AlternativeAssessmentDescription, ExpertDescription, \
    AlternativeDescription, AlternativeAssessmentForSingleCriteriaDescription, CriterionDescription, ScalesDescription


def generate_assessments(criteria, alternatives, experts, scales, task_type: TaskType) \
        -> dict[str, list[AlternativeAssessmentDescription]]:
    numeric_criteria_seeds = {}
    all_assessments = {}
    for expert_info in experts:
        expert_info: ExpertDescription
        alternatives_assessments: list[AlternativeAssessmentDescription] = []

        for alternative_info in alternatives:
            alternative_info: AlternativeDescription
            alternative_assessment: list[AlternativeAssessmentForSingleCriteriaDescription] = []
            for _, criteria_values in criteria.items():

                for criterion_info in criteria_values:
                    criterion_info: CriterionDescription

                    if criterion_info.qualitative and task_type is TaskType.NUMERIC_ONLY:
                        raise ValueError('unable to create linguistic variables for numeric only tasks')
                    elif criterion_info.qualitative and task_type is TaskType.HYBRID_CRISP_LINGUISTIC:
                        # IMPORTANT: hardcoded value of crisp scale as it is the only one to the moment
                        scale: ScalesDescription = list(filter(lambda x: x.scaleID == 's5', scales))[0]
                        scale_id = scale.scaleID
                        value = random.choice(scale.labels)
                    elif criterion_info.qualitative and task_type is TaskType.HYBRID_FUZZY_LINGUISTIC:
                        scale: ScalesDescription = random.choice(scales)
                        scale_id = scale.scaleID
                        value = random.choice(scale.labels)
                    elif not criterion_info.qualitative:
                        if numeric_criteria_seeds.get(criterion_info.criteriaID) is None:
                            random_seed = random.randint(1, 5) * 10 * random.randint(1, 3)
                            numeric_criteria_seeds[criterion_info.criteriaID] = random_seed
                        random_seed = numeric_criteria_seeds[criterion_info.criteriaID]
                        scale_id = None
                        value = round(random.uniform(random_seed * 0.5, random_seed * 1.5), 2)
                        should_round_to_int = random.choice([True, False])
                        if should_round_to_int:
                            value = round(value)

                    new_assessment = AlternativeAssessmentForSingleCriteriaDescription(
                        criteriaID=criterion_info.criteriaID,
                        estimation=[str(value), ],
                        scaleID=scale_id)
                    alternative_assessment.append(new_assessment)
            alternatives_assessments.append(AlternativeAssessmentDescription(
                alternativeID=alternative_info.alternativeID,
                criteria2Estimation=alternative_assessment
            ))
        all_assessments[expert_info.expertID] = alternatives_assessments
    return all_assessments
