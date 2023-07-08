# DSR-LM

This is the implementation corresponding to the paper [Improved Logical Reasoning of Language Models via Differentiable Symbolic Programming](https://arxiv.org/pdf/2305.03742.pdf).


## Setup
To setup the environment running this project, we provide a conda yml file. You can install the environment by running the following code:
```
conda env create --name dsrlm -f environment.yml
conda activate dsrlm
python -m pip install ./scallopy-0.1.9-cp310-cp310-manylinux_2_31_x86_64.whl
```
Note that the wheel is provided only for linux platform, if you are running the code on another system, you can build the [scallopy package](https://github.com/Liby99/scallop-v2) from scratch.

## Run the script
You can train your own model by running the following code:
```
python ./clutrr/run_with_constraints.py
```
