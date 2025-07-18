# Enhancing Visual Reliance in Text Generation: A Bayesian Perspective on Mitigating Hallucination in Large Vision-Language Models[[Paper](https://arxiv.org/abs/2505.19498)]


## Overview

<p align="center">
    <img src="https://github.com/NeilHnxTcc/EVRB/blob/main/imgs/evrb.png" width="90%"></a> <br>
</p>


## Setup
```
conda create -n evrb python==3.10
conda activate evrb
python install -r requirements.txt
python install qwen-vl-utils
cd vendor
pip install tokenizers==0.13.3 --no-deps --target=./
pip install sentencepiece --no-deps --target=./
```


## Downloads

MSCOCO 2014 / AOKVQA / GPA / Visual Genome dataset are needed. Please download [here](https://cocodataset.org/#home) 

You need to prepare the following checkpoints of base models:
- Download [LLaVA-1.5-7b](https://huggingface.co/liuhaotian/llava-v1.5-7b/tree/main).
- Download [CLIP](https://huggingface.co/openai/clip-vit-large-patch14-336).
- Download [Qwen2.5-vl](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct).


## Implementation
After setup the environment, you can directly use our code base to imply EVRB on <b>llava-1.5-7b</b> and <b>Qwen2.5-vl-7b</b> and evalute the performance on three benchmarks: <b>POPE</b>, <b>CHAIR</b>, and <b>MME</b>: 


llava pope
```
nohup python scripts/llava_1_5_7b/pope_evrb.py --do-ct --gpu-id 0 &>"./outputs/pope_llava_evrb.log" &
nohup python scripts/llava_1_5_7b/pope_evrb.py --gpu-id 0 &>"./outputs/pope_llava_vanilla.log" & 
```
llava chair 
```
nohup python scripts/llava_1_5_7b/chair_evrb.py --do-ct --do-eos --gpu-id 0 --save-name chair_evrb_eager --save-folder outputs/ &>"outputs/tempt.log" &
nohup python scripts/llava_1_5_7b/chair_evrb.py --gpu-id 2 --save-name chair_vanilla_eager --save-folder outputs/ &>"outputs/tempt.log" &
```

llava mme 
```
nohup python scripts/llava_1_5_7b/mme_evrb.py  --gpu-id 2 --do-ct  --save-folder ./outputs/ --save-name "mme_llava_evrb"  &>"outputs/tempt_1.log" &
nohup python scripts/llava_1_5_7b/mme_evrb.py  --gpu-id 1  --save-folder ./outputs/ --save-name "mme_llava_vanilla"  &>"outputs/tempt_1.log" &
```




qwen pope
```
nohup python scripts/qwen_2_5_vl_7b/pope_evrb.py --do-ct --gpu-id 0 &>"./outputs/pope_qwen_evrb.log" &
nohup python scripts/qwen_2_5_vl_7b/pope_evrb.py --gpu-id 1 &>"./outputs/pope_qwen_vanilla.log" & 
```

qwen chair
```
nohup python scripts/qwen_2_5_vl_7b/chair_evrb.py --do-ct --do-eos --gpu-id 2 --save-name chair_evrb --save-folder outputs/ &>"outputs/tempt_1.log" &
nohup python scripts/qwen_2_5_vl_7b/chair_evrb.py --gpu-id 4 --save-name chair_vanilla --save-folder outputs/ &>"outputs/tempt_1.log" &
```

qwen mme 
```
nohup python scripts/qwen_2_5_vl_7b/mme_evrb.py  --gpu-id 5 --do-ct  --save-folder ./outputs/ --save-name "mme_qwen_evrb"  &>"outputs/tempt_1.log" &
nohup python scripts/qwen_2_5_vl_7b/mme_evrb.py  --gpu-id 6  --save-folder ./outputs/ --save-name "mme_qwen_vanilla"  &>"outputs/tempt_1.log" &
```


Additionally, running the upper codes, POPE can directly output the evaluation metrics, while CHAIR and MME output the generated texts. We need to run additional code to get the result:


# chair
```
python utils/eval_chair.py --cap_file <jsonl file path for captions>
# mme
python utils/eval_mme.py --results_dir <folder path for mme results>

```

## Acknowledgement
Some codes are based on the [Qwen2.5-vl](https://github.com/QwenLM/Qwen2.5-VL) and [SID](https://github.com/huofushuo/SID). Thanks for their excellent work!


## Citation
If you find this work useful for your research, please cite [our paper](https://arxiv.org/abs/2505.19498):
```
@article{hu2025enhancing,
  title={Enhancing Visual Reliance in Text Generation: A Bayesian Perspective on Mitigating Hallucination in Large Vision-Language Models},
  author={Hu, Nanxing and Duan, Xiaoyue and Zhang, Jinchao and Kang, Guoliang},
  journal={arXiv preprint arXiv:2505.19498},
  year={2025}
}
```







