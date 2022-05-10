set -ex

which python

python -m pip install --upgrade pip
python -m pip install virtualenv
python -m virtualenv venv

source venv/bin/activate

which python

python -m pip install -r requirements.txt
python -m pip install -r requirements_qa.txt

curl -L -o lingvo-dss-all.jar https://github.com/ldss-hse/ldss-core-aggregator/releases/download/decision_maker_v0.7/lingvo-dss-all.jar
mv lingvo-dss-all.jar benchmark/methods/ml_ldm/scripts/bin/