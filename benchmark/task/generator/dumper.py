import json
from pathlib import Path

# pylint: disable=no-name-in-module
from pydantic.json import pydantic_encoder


def save_to_json(path: Path, res_dto):
    with path.open('w', encoding='utf-8') as file:
        json.dump(res_dto, file, indent=4, default=pydantic_encoder)
