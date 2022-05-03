import json
from pathlib import Path

from pydantic.json import pydantic_encoder

from benchmark.task.schemas.task_scheme import TaskDTOScheme


def save_to_json(path: Path, res_dto: TaskDTOScheme):
    with path.open('w', encoding='utf-8') as file:
        json.dump(res_dto, file, indent=4, default=pydantic_encoder)
