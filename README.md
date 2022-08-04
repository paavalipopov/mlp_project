# Experiment Template

- `.github/workflow` - codestyle and CI
- `assets` - datasets, logs, etc
- `bin` - bash files to start pipelines
- `docker` - docker files
- `examples` - notebooks and full-featured examples
- `requirements` - python requirements
- `src` - code
- `tests` - tests

## How to reproduce?

```bash
bash bin/...  # download data
pip install -r ./requirements/...  # install dependencies, or use docker
bash bin/...  # run experiments
# use examples/... to analize results
```

## Example
PYTHONPATH=./ python src/scripts/tune_ts_dl.py --model mlp --ds fbirn --max-epochs 100 --num-trials 10

PYTHONPATH=./ python src/scripts/tune_ts_dl_parallel.py --model mlp --ds fbirn --max-epochs 100 --num-trials 10

PYTHONPATH=./ python src/scripts/tune_ts_dl.py --model mlp --ds abide_869 --max-epochs 30 --num-trials 1

PYTHONPATH=./ python src/scripts/lstm_oasis.py --model mlp --ds abide_869 --max-epochs 100 --num-trials 10

PYTHONPATH=./ python src/scripts/lstm_oasis.py --model mlp --ds fbirn --max-epochs 100 --num-trials 10

PYTHONPATH=./ python src/scripts/tune_ts_dl.py --model mlp --ds ukb --max-epochs 200 --num-trials 20

PYTHONPATH=./ python src/scripts/tune_ts_baselines.py --model lr --ds fbirn --num-trials 10

PYTHONPATH=./ python src/scripts/tune_ts_dl.py --model ens-lr --ds fbirn --max-epochs 200 --num-trials 10