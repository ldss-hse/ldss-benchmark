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
        [2, "[{'m': 0.21458054}]"],
        [0, "[{'m': 0.16432953}]"],
        [3, "[{'m': -0.24468136}]"],
        [1, "[{'m': -0.37298417}]"]
    ]
    expected_res = np.array(expected_res_raw, dtype=object)
    assert np.array_equal(res, expected_res), 'Reported aggregation results do not match'
