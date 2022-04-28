set -ex

source venv/bin/activate

python -m pytest

echo "Tests passed."
