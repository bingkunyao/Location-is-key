These are open-sourced materials of the paper "Location is Key: Leveraging LLM for Functional Bug Localization in Verilog Design". 

## Directory Structure

testset/: The test dataset of LiK.

requirements.txt: Specify the required dependencies.

inference_example.py: Provide an example for bug localization in module "adder_8bit".

## Model Usage

Firstly, you need to install all dependencies using 
```python
pip install -r requirements.txt
```

Then, run inference_example.py, which provides an example for LiK inference. 

## Link

The LiK model can be downloaded from https://huggingface.co/zwhc/LiK.

The training data can be found at https://huggingface.co/datasets/zwhc/LiK_train_dataset.









