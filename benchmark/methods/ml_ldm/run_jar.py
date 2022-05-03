import enum
import json
import platform
import subprocess
from pathlib import Path

from benchmark.constants import ARTIFACTS_PATH


class DecisionMakerCrashedError(Exception):
    pass


class CLIExecutableEnum(enum.Enum):
    bash = 1
    powershell = 2
    python = 3

    def __str__(self):
        mapping = {
            CLIExecutableEnum.bash: 'bash',
            CLIExecutableEnum.python: 'python',
            CLIExecutableEnum.powershell: 'Powershell.exe',
        }
        return mapping[self]


def _run_console_tool(tool_path: Path, exe: CLIExecutableEnum = CLIExecutableEnum.bash, *args, **kwargs):
    kwargs_processed = []
    for item in kwargs.items():
        if item[0] in ('env', 'debug'):
            continue
        kwargs_processed.extend(map(str, item))

    options = [str(exe)]
    if exe is CLIExecutableEnum.powershell:
        options.append('-File')

    options.extend([
        str(tool_path),
        *args,
        *kwargs_processed
    ])

    if kwargs.get('debug', False):
        print(f'Attempting to run with the following arguments: {" ".join(options)}')

    if kwargs.get('env'):
        return subprocess.run(options, capture_output=True, env=kwargs.get('env'))
    return subprocess.run(options, capture_output=True)


def _run_jar(scripts_path: Path, job_artifacts_path: Path, task_json_path: Path):
    if platform.system() == 'Windows':
        tool_path = scripts_path / 'run_decision_maker.ps1'
        exe = CLIExecutableEnum.powershell
    else:
        tool_path = scripts_path / 'run_decision_maker.sh'
        exe = CLIExecutableEnum.bash

    jar_path = scripts_path / 'bin' / 'lingvo-dss-all.jar'

    arguments = [
        '-JAR_PATH', str(jar_path),
        '-INPUT_JSON', str(task_json_path),
        '-OUTPUT_DIR', str(job_artifacts_path)
    ]
    res_process = _run_console_tool(tool_path, exe, *arguments, debug=True)
    stdout = str(res_process.stdout.decode("utf-8"))
    print(f'SUBPROCESS: {stdout}')
    print(f'SUBPROCESS: {str(res_process.stderr.decode("utf-8"))}')

    if res_process.returncode != 0 or '[ERROR] ' in stdout:
        raise DecisionMakerCrashedError('Decision Maker did not finish successfully')


def parse_results(json_path: Path):
    print('In parsing results')
    with json_path.open(encoding='utf-8') as file:
        res = json.load(file)
    return res


def run_decision_maker(task_id: int, task_json_path: Path):
    parent_path = Path(__file__).parent
    scripts_path = parent_path / 'scripts'
    job_artifacts_path = ARTIFACTS_PATH / str(task_id)

    _run_jar(scripts_path=scripts_path, job_artifacts_path=job_artifacts_path, task_json_path=task_json_path)

    json_path = job_artifacts_path / 'result.json'
    parsed_results = parse_results(json_path)

    return parsed_results
