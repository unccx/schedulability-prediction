# schedulability-prediction


# Installation

## conda/mamba
```bash
mamba env create -f .\environment.yml
conda activate task_hg
pip install git+https://github.com/unccx/DeepHypergraph.git --no-deps
pip install git+https://github.com/unccx/simRT.git
```

## poetry
```shell
poetry install
poetry shell
```