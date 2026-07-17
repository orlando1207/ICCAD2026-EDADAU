# MacroDiff: A Geometric Diffusion Model for Macro Placement Generation
This is the repository for MacroDiff, macro placement framework using diffusion model.

## Publication
Jongho Yoon, Jinsung Jeon, and Seokhyeong Kang, "Late Breaking Results: A Geometric Diffusion Model for Macro Placement Generation", in Proc. DAC, 2025.


## Requirements
(Other versions may work, but not tested)
- Python == 3.9
- PyTorch == 2.1.2
- PyG == 2.5.3
- Other dependencies (see `requirements.txt`)

Install dependencies with:
```bash
pip install -r requirements.txt
```

**Note**: To use the CUDA version of `pytorch-scatter`, you need to install it seperately.
Run the following command, where `${CUDA}` is your CUDA version (e.g., `cu121`):
```bash
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.2+${CUDA}.html
```


## Training
Due to size constraints, this repository includes only 8 *ISPD2005* datasets. 
You can use these datasets to test the training process.

Run training with:
```bash
python train.py
```

## Sampling
Running the sampling process yields macro layouts. 
A pretrained weight is provided at `checkpoint/checkpoint.ckpt`. 

To run sampling with the pretrained model, use the following command:

```bash
python test.py -c ./checkpoint/checkpoint.ckpt
```
