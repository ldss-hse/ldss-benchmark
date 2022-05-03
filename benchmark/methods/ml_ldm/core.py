import random
import re
from pathlib import Path

import numpy as np
import wget

from benchmark.constants import ML_LDM_JAR_RELEASE_LINK
from benchmark.methods.common.idecision_maker import IDecisionMaker
from benchmark.methods.ml_ldm.result_model import MLLDMTaskResultDTOScheme, ResultAlternativeAssessmentDescription
from benchmark.methods.ml_ldm.run_jar import run_decision_maker


class MLLDMDecisionMaker(IDecisionMaker):
    def run(self):
        artifact_id = random.randrange(1000)
        print(f'ML-LDM artifact id: {artifact_id}')

        jar_folder = Path(__file__).parent / 'scripts' / 'bin'
        jar_folder.mkdir(parents=True, exist_ok=True)

        jar_path = jar_folder / 'lingvo-dss-all.jar'

        if not jar_path.exists():
            print(f'Binary JAR for ML-LDM is not present on this machine. Installing it first...')
            wget.download(ML_LDM_JAR_RELEASE_LINK, str(jar_path))
            print(f'Installed!')

        task_result_raw = run_decision_maker(artifact_id, self._task.json_path)

        # pylint: disable=no-member
        task_parsed_dto = MLLDMTaskResultDTOScheme(**task_result_raw)
        task_parsed_dto: MLLDMTaskResultDTOScheme

        # result is an array of elements [alternative_id, assessment]
        res = np.empty((self._task.num_alternatives, 2), dtype=object)
        id_pattern = re.compile(r'(\d+)')
        for alternative_index, alternative in enumerate(task_parsed_dto.alternativesOrdered):
            alternative: ResultAlternativeAssessmentDescription

            id_match = id_pattern.search(alternative.alternativeID)
            if id_match is None:
                raise ValueError('Alternative ID does not contain number')

            # IDs in JSON description start from 1
            res[alternative_index][0] = int(id_match.group(0)) - 1
            res[alternative_index][1] = str(alternative.estimation)

        return res
