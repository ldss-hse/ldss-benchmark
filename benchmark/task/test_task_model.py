import numpy as np

from benchmark.constants import TASKS_ROOT
from benchmark.task.task_model import TaskModelFactory


def test_task_model_aircraft():
    path_to_task = TASKS_ROOT / '1_aircraft' / 'task.json'
    task = TaskModelFactory().from_json(path_to_task)

    raw_data = [
        [2., 1500, 20000, 5.5, 5, 9],
        [2.5, 2700, 18000, 6.5, 3, 5],
        [1.8, 2000, 21000, 4.5, 7, 7],
        [2.2, 1800, 20000, 5.0, 5, 5],
    ]
    expected_matrix = np.array(raw_data)
    assert len(task._decision_matrices) == 1, 'JSON contains only 1 matrix'
    assert np.array_equal(task._decision_matrices['expert1'].get_raw(), expected_matrix), 'Load from JSON failed'
