# `kornia-vlm` validator

This is a fork of the original VLMEvalKit: [repo link](https://github.com/open-compass/VLMEvalKit).


## Guides

Building the environment
```sh
cd vlmeval/vlm_custom/kornia_vlm_pyo3
# create your conda/venv/uv environment here

# for venv
source <name>/bin/activate

# make sure to deactivate any existing environment (in this, conda environments)
conda deactivate

# build your rust-python interop
maturin develop --release
```

Running the validation commands
```sh
# Initialize the environment if haven't yet

source ./vlmeval/vlm_custom/kornia_vlm_pyo3/.env/bin/activate

# for running the Kornia's version
python run.py --data <dataset> --model SmolVLM-Kornia --reuse

# for running the Python's version with the same feature
python run.py --data <dataset> --model SmolVLM-Py-Comparison --reuse

# for running the Python's default features (rarely used)
python run.py --data <dataset> --model SmolVLM
```

[Link](https://aicarrier.feishu.cn/wiki/Qp7wwSzQ9iK1Y6kNUJVcr6zTnPe?table=tblsdEpLieDoCxtb&view=vewa8sGZrY) to their supported dataset name.
