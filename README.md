# `kornia-vlm` validator

This is a fork of the original VLMEvalKit: [repo link](https://github.com/open-compass/VLMEvalKit).

Custom interop & eval code are in `./vlmeval/vlm_custom`.

## Guides

Building & restarting the environment. Make sure to check your path to kornia is correct in `vlmeval/vlm_custom/kornia_vlm_pyo3/Cargo.toml`.
```sh
# create environment
source .venv/bin/activate   # for example, I'm using venv

# VLMEvalKit setup
pip install -e .

# maturin setup
pip install maturin


# build your rust-python interop
maturin develop --release -m vlmeval/vlm_custom/kornia_vlm_pyo3/Cargo.toml
```

Running the validation commands.
```sh
# Initialize the environment if haven't yet
source .venv/bin/activate   # for example, I'm using venv

# for running the Kornia's version
python run.py --data <dataset> --model SmolVLM-Kornia --reuse

# for running the Python's version with the same feature
python run.py --data <dataset> --model SmolVLM-Py-Comparison --reuse

# for running the Python's default features (rarely used)
python run.py --data <dataset> --model SmolVLM --reuse


python run.py --data <dataset> --model SmolVLM2-Kornia --reuse
python run.py --data <dataset> --model SmolVLM2-Py-Comparison --reuse
```

[Link](https://aicarrier.feishu.cn/wiki/Qp7wwSzQ9iK1Y6kNUJVcr6zTnPe?table=tblsdEpLieDoCxtb&view=vewa8sGZrY) to their supported dataset name.
