import pytest

from benchmark.constants import ARTIFACTS_PATH
from benchmark.methods.electre_i.core import ElectreDecisionMaker
from benchmark.methods.ml_ldm.core import MLLDMDecisionMaker
from benchmark.methods.topsis.core import TopsisDecisionMaker
from benchmark.task.generator.dumper import save_to_json
from benchmark.task.generator.single_task_generator import SingleTaskGenerator
from benchmark.task.generator.task_type import TaskType
from benchmark.task.schemas.task_scheme import TaskDTOScheme
from benchmark.task.task_model import TaskModelFactory


@pytest.mark.skip(reason='ML-LDM does not currently support explicit expert weights')
def test_end_to_end_single_generator_numeric_task():
    num_alternatives = 4
    generator = SingleTaskGenerator(num_experts=1,
                                    num_alternatives=num_alternatives,
                                    num_criteria_groups=1,
                                    num_criteria_per_group=6,
                                    task_type=TaskType.NUMERIC_ONLY)
    res_dto: TaskDTOScheme = generator.run()

    path = ARTIFACTS_PATH / 'tests_artifacts' / 'gen_task_1.json'
    path.parent.mkdir(exist_ok=True, parents=True)
    save_to_json(path, res_dto)

    task = TaskModelFactory().from_json(path)

    decision_maker: ElectreDecisionMaker = ElectreDecisionMaker(task)
    res = decision_maker.run()
    assert len(res) == num_alternatives, 'All alternatives should be present in ranking'

    decision_maker: TopsisDecisionMaker = TopsisDecisionMaker(task)
    res = decision_maker.run()
    assert len(res) == num_alternatives, 'All alternatives should be present in ranking'

    decision_maker: MLLDMDecisionMaker = MLLDMDecisionMaker(task)
    res = decision_maker.run()
    assert len(res) == num_alternatives, 'All alternatives should be present in ranking'
