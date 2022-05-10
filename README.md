# ldss-benchmark

## Running benchmark

1. Prerequisites:
   1. Python 3.9
   2. Java (OpenJDK) 17
2. Configure project (install all Python dependencies)
3. Download compliant ML-LDM binary release (currently supported `v0.7`) from 
   [releases page](https://github.com/ldss-hse/ldss-core-aggregator/releases)
   and put it into `./benchmark/methods/ml_ldm/scripts/bin/lingvo-dss-all.jar`
4. Run benchmarking experiments, for example: `python benchmark/comparison/experiment_1.py`
5. All experiment results are visualized and placed in an artifacts directory, for example in
   `artifacts/generated_tasks/experiment_1/report/visualization`

## Benchmarking experiments

### Experiment  no. 1

| Parameter              | Value           |
|:-----------------------|:----------------|
| Number of experts      | 1               |
| Weights of experts     | Equal           |
| Number of alternatives | (3, 5, 7, 9)    |
| Number of criteria     | (5, 10, 15, 20) |
| Types of assessments   | Numeric         |

### Experiment  no. 2

| Parameter              | Value                     |
|:-----------------------|:--------------------------|
| Number of experts      | 10                        |
| Weights of experts     | Equal                     |
| Number of alternatives | (3, 5, 7, 9)              |
| Number of criteria     | (5, 10, 15, 20)           |
| Types of assessments   | Numeric, Crisp Linguistic |

### Experiment  no. 3

| Parameter              | Value                     |
|:-----------------------|:--------------------------|
| Number of experts      | 10                        |
| Weights of experts     | Automatically assigned    |
| Number of alternatives | (3, 5, 7, 9)              |
| Number of criteria     | (5, 10, 15, 20)           |
| Types of assessments   | Numeric, Crisp Linguistic |
