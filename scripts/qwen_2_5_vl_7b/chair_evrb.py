import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm



from datetime import datetime


import json
import warnings
warnings.filterwarnings("ignore")



from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


from utils.evrb_sample import evolve_my_sampling
evolve_my_sampling()
from utils.evrb_qwen import use_my_qwen
use_my_qwen()

from utils.hyper_config import hyper_param


time = datetime.now().strftime('%m-%d-%H:%M')
print(time)



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
parser.add_argument("--data-path", type=str, default="/mnt/gemininjceph2/geminicephfs/wx-mm-spr-xxxx/neilnxhu/datasets/coco/val2014", help="data path")
parser.add_argument("--batch-size", type=int, default=1, help="batch size")
parser.add_argument("--num-workers", type=int, default=1, help="num workers")


parser.add_argument("--test-sample", type=int, default=500)
parser.add_argument("--threshold", type=int, default=15)
parser.add_argument("--save-name", type=str, required=True)
parser.add_argument("--save-folder", type=str, required=True)

# my args
parser.add_argument("--img-ent-thr", type=float, default=6.5)
parser.add_argument("--pri-rec-thr", type=float, default=0.9)
parser.add_argument("--do-ct", action='store_true', default=False)
parser.add_argument("--do-eos", action='store_true', default=False)
parser.add_argument('---eos-k', type=float, default=1.5)
parser.add_argument('--vv_thr', type=float, default=0.05)
args = parser.parse_args()

hyper_param.img_ent_thr = args.img_ent_thr
hyper_param.pri_rec_thr = args.pri_rec_thr
hyper_param.do_ct = args.do_ct
hyper_param.do_eos = args.do_eos
hyper_param.eos_k = args.eos_k
hyper_param.vv_thr = args.vv_thr
hyper_param.img_id = 151655
hyper_param.stop_id = 13



args = parser.parse_args()


os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
setup_seeds()
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

# ========================================
#             Model Initialization
# ========================================
print('Initializing Model')


messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "",
            },
            {"type": "text", "text": ""},
        ],
    }
]

ckpt_path = "/mnt/gemininjceph2/geminicephfs/wx-mm-spr-xxxx/neilnxhu/Qwen2.5-VL/Qwen/Qwen2.5-VL-7B-Instruct"

if args.do_eos:
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    ckpt_path, torch_dtype="auto", device_map=device, attn_implementation='eager',
) 
else:    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        ckpt_path, torch_dtype="auto", device_map=device
    )    

model.eval()

processor = AutoProcessor.from_pretrained(ckpt_path)



img_files = os.listdir(args.data_path)
random.shuffle(img_files)

with open('/mnt/gemininjceph2/geminicephfs/wx-mm-spr-xxxx/neilnxhu/datasets/coco/annotations/instances_val2014.json', 'r') as f:
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


# base_dir  = "./log/" + args.model
base_dir = args.save_folder
if not os.path.exists(base_dir):
    os.makedirs(base_dir)



save_path = os.path.join(base_dir, 'qwen-{}-{}.jsonl'.format(args.test_sample, args.save_name))
if os.path.exists(save_path):
    os.remove(save_path)

for img_id in tqdm(range(len(img_files))):
# for img_id in range(len(img_files)):
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

    qu = "Please describe this image in detail."

    messages[0]['content'][0]["image"] = image_path
    messages[0]['content'][1]["text"] = qu
    
    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False, output_attentions=args.do_eos)


    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )           

        
    img_save["caption"] = output_text


    with open(save_path, "a") as f:
                json.dump(img_save, f)
                f.write('\n')


    


