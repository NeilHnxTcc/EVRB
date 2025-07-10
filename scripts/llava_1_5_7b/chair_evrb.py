import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm





from PIL import Image
from datetime import datetime

import json
import warnings
warnings.filterwarnings("ignore")



from transformers import LlavaForConditionalGeneration, AutoProcessor


# from utils.evrb_sample import evolve_my_sampling
from utils.evrb_llava_sample import evolve_my_sampling
from utils.evrb_llava import use_my_llava

from utils.hyper_config import hyper_param


time = datetime.now().strftime('%m-%d-%H:%M')
print(time)


import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


MODEL_EVAL_CONFIG_PATH = {
    "llava-1.5": "eval_configs/llava-1.5_eval.yaml",
    "llava-1.5-13b": "eval_configs/llava-1.5_eval_13b.yaml",
}


def setup_seeds():
    seed =  42

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


parser = argparse.ArgumentParser(description="POPE-Adv evaluation on LVLMs.")
parser.add_argument("--gpu-id", type=int, default=2, help="specify the gpu to load the model.")
# vision contrastive decoding
parser.add_argument(
    "--options",
    nargs="+",
    help="override some settings in the used config, the key-value pair "
    "in xxx=yyy format will be merged into config file (deprecate), "
    "change to --cfg-options instead.",
)
parser.add_argument("--data-path", type=str, default="../datasets/coco/val2014", help="data path")
parser.add_argument("--batch-size", type=int, default=1, help="batch size")
parser.add_argument("--num-workers", type=int, default=1, help="num workers")


parser.add_argument("--test-sample", type=int, default=500)
parser.add_argument("--save-name", type=str, required=True)
parser.add_argument("--save-folder", type=str, required=True)

# my args
parser.add_argument("--img-ent-thr", type=float, default=7.48)
parser.add_argument("--pri-rec-thr", type=float, default=0.9)
parser.add_argument("--do-ct", action='store_true', default=False)
parser.add_argument("--do-eos", action='store_true', default=False)
parser.add_argument('---eos-k', type=float, default=1.5)
parser.add_argument('--vv_thr', type=float, default=0.05)
args = parser.parse_args()
args = parser.parse_args()
if args.do_ct:
    evolve_my_sampling()
if args.do_eos:
    use_my_llava()
hyper_param.img_ent_thr = args.img_ent_thr
hyper_param.pri_rec_thr = args.pri_rec_thr
hyper_param.do_ct = args.do_ct
hyper_param.do_eos = args.do_eos
hyper_param.eos_k = args.eos_k
hyper_param.vv_thr = args.vv_thr
hyper_param.img_id = 32000
hyper_param.stop_id = 29889
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
setup_seeds()



# ========================================
#             Model Initialization
# ========================================
print('Initializing Model')


ckpt_path = "./ckpts/llava-v1.5-7b-hf"

model = LlavaForConditionalGeneration.from_pretrained(ckpt_path, torch_dtype="auto", device_map=device,) 

  





print("Done!")


processor = AutoProcessor.from_pretrained(ckpt_path)



# mean = (0.48145466, 0.4578275, 0.40821073)
# std = (0.26862954, 0.26130258, 0.27577711)
# norm = transforms.Normalize(mean, std)


img_files = os.listdir(args.data_path)
random.shuffle(img_files)

with open('../datasets/coco/annotations/instances_val2014.json', 'r') as f:
    lines = f.readlines()
coco_anns = json.loads(lines[0])

img_dict = {}

categories = coco_anns["categories"]
category_names = [c["name"] for c in categories]
category_dict = {int(c["id"]): c["name"] for c in categories}

for img_info in coco_anns["images"]:
    img_dict[img_info["id"]] = {"name": img_info["file_name"], "anns": []}

for ann_info in coco_anns["annotations"]:
    img_dict[ann_info["image_id"]]["anns"].append(
        category_dict[ann_info["category_id"]]
    )


base_dir  = args.save_folder
if not os.path.exists(base_dir):
    os.makedirs(base_dir)


    
    
save_path = os.path.join(base_dir, 'llava-{}-{}.jsonl'.format(args.test_sample, args.save_name))
if os.path.exists(save_path):
    os.remove(save_path)


for img_id in tqdm(range(len(img_files))):
# for img_id in range(86,len(img_files)):
    if img_id == args.test_sample:
        break
    img_file = img_files[img_id]
    img_id = int(img_file.split(".jpg")[0][-6:])
    img_info = img_dict[img_id]
    assert img_info["name"] == img_file
    img_anns = set(img_info["anns"])
    img_save = {}
    img_save["image_id"] = img_id

    image_path = args.data_path + "/" + img_file
    raw_image = Image.open(image_path).convert("RGB")

    qu = "Please describe this image in detail."

    conversation = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions." }
            ]
        },

        {
        "role": "user",
        "content": [
            {"type": "text", "text": qu},
            {"type": "image"},
            ],
        },
    ]

    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    inputs = processor(images=raw_image, text=prompt, return_tensors='pt').to(0, torch.float16)

    
    with torch.inference_mode():
        with torch.no_grad():
            output  = model.generate(
                **inputs,
                max_new_tokens = 512,
                attn_implementation="eager",
            )
    start_idx = inputs['input_ids'].shape[-1]
    output_text = processor.decode(output[0][start_idx:], skip_special_tokens=True)
    img_save["caption"] = output_text
    

    with open(save_path, "a") as f:
                json.dump(img_save, f)
                f.write('\n')
               
                
                
                
    # # dump metric file
    # if args.use_fast_v == True and args.use_cd == True:
    #     with open(os.path.join(base_dir, 'top_important_ours-{}samples-cd-layer{}-token{}-time{}-greedy.jsonl'.format(args.test_sample, args.fast_v_agg_layer, args.fast_v_attention_rank, time)), "a") as f:
    #         json.dump(img_save, f)
    #         f.write('\n')
    # elif args.use_vcd == True:
    #     with open(os.path.join(base_dir, 'degraded_ours-{}samples-vcd-sampling.jsonl'.format(args.test_sample)), "a") as f:
    #         json.dump(img_save, f)
    #         f.write('\n')
    # elif args.use_icd == True:
    #     with open(os.path.join(base_dir, 'degraded_ours-{}samples-icd-sampling.jsonl'.format(args.test_sample)), "a") as f:
    #         json.dump(img_save, f)
    #         f.write('\n')
    # else:
    #     with open(os.path.join(base_dir, 'ours-{}samples-opera.jsonl'.format(args.test_sample)), "a") as f:
    #         json.dump(img_save, f)
    #         f.write('\n')

    


