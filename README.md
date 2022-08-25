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

## Examples
for model in mlp new_attention_mlp lstm transformer
do
    for dataset in oasis abide fbirn cobre abide_869
    do
        PYTHONPATH=./ python src/scripts/ts_dl_experiments.py --mode tune --model $model --ds $dataset --max-epochs 200 --num-trials 10     
        PYTHONPATH=./ python src/scripts/ts_dl_experiments.py --mode experiment --model $model --ds $dataset --max-epochs 200 --num-trials 10     
    done
done
