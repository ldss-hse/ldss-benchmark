from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
TASKS_ROOT = PROJECT_ROOT / 'tasks'
ARTIFACTS_PATH = PROJECT_ROOT / 'artifacts'
GENERATED_TASKS_PATH = ARTIFACTS_PATH / 'generated_tasks'
ML_LDM_JAR_RELEASE_VERSION = 'decision_maker_v0.5'
ML_LDM_REPO_LINK = 'https://github.com/ldss-hse/ldss-core-aggregator'
ML_LDM_JAR_RELEASE_LINK = f'{ML_LDM_REPO_LINK}/releases/download/{ML_LDM_JAR_RELEASE_VERSION}/lingvo-dss-all.jar'
