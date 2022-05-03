import numpy as np

from benchmark.constants import TASKS_ROOT
from benchmark.methods.ml_ldm.core import MLLDMDecisionMaker
from benchmark.task.task_model import TaskModelFactory


def test_aircraft_selection_ml_ldm():
    path_to_task = TASKS_ROOT / '1_aircraft' / 'task.json'
    task = TaskModelFactory().from_json(path_to_task)

    decision_maker: MLLDMDecisionMaker = MLLDMDecisionMaker(task)

    res = decision_maker.run()

    expected_res_raw = [
        [0, "[{'m': 0.17255354}]"],
        [2, "[{'m': 0.11221862}]"],
        [1, "[{'m': -0.25417423}]"],
        [3, "[{'m': -0.29175043}]"]
    ]
    expected_res = np.array(expected_res_raw, dtype=object)
    assert np.array_equal(res, expected_res), 'Reported aggregation results do not match'
