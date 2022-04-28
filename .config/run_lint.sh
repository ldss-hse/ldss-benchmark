set -ex

echo -e '\n'
echo 'Running lint check...'

source venv/bin/activate

python -m pylint benchmark

echo "Pylint passed."
