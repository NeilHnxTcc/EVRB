
#-------
import sys
sys.path.insert(0, "vendor")
sys.path.insert(0,"vendor/transformers/src")
#-------
import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm

from torchvision import transforms


from PIL import Image

from datetime import datetime


import json
import warnings
warnings.filterwarnings("ignore")

from utils.evrb_llava import LLaVa
from utils.evrb_llava_sample import evolve_my_sampling
from utils.evrb_llava import use_my_llava
from utils.hyper_config import hyper_param
from omegaconf import OmegaConf
time = datetime.now().strftime('%m-%d-%H:%M')
print(time)


import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'




def build_model_config(config, default_config = './utils/llava/config/llava-1.5_vicuna7b.yaml'):
    model_config_path = default_config
    model_config = OmegaConf.create()
    model_config = OmegaConf.merge(
        model_config,
        OmegaConf.load(model_config_path),
        {"model": config["model"]},
    )
    return model_config

def build_preprocessor_config(default_config = './utils/llava/config/llava-1.5_vicuna7b.yaml'):
    config = OmegaConf.load(default_config)
    return config['preprocess']


from utils.llava.clip_processors import ClipImageEvalProcessor
 


def setup_seeds():
    seed = 42

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True



parser = argparse.ArgumentParser(description="CHAIR evaluation on LVLMs.")
parser.add_argument("--model", type=str, default="llava-1.5", help="model")
parser.add_argument("--gpu-id", type=str,  default='0', help="specify the GPUs to load the model.")
parser.add_argument("--data-path", type=str, default="../datasets/coco/val2014", help="data path")
parser.add_argument("--cfg-path", type=str, default="./utils/llava/config/llava-1.5_eval.yaml", help="cfg path")
parser.add_argument("--batch-size", type=int, default=1, help="batch size")
parser.add_argument("--num_workers", type=int, default=1, help="num workers")
parser.add_argument("--answers-file", type=str, default="")

parser.add_argument("--sample-greedy", action='store_true',  default=True)
parser.add_argument("--sample", action='store_true',  default=True)
parser.add_argument("--options")


parser.add_argument("--test-sample", type=int, default=500)
parser.add_argument("--save-name", type=str, required=True) 
parser.add_argument("--save-folder", type=str, default='outputs/')

# my args
parser.add_argument("--img-ent-thr", type=float, default=7.48)
parser.add_argument("--pri-rec-thr", type=float, default=0.9)
parser.add_argument("--do-ct", action='store_true', default=False)
parser.add_argument("--do-eos", action='store_true', default=False)
parser.add_argument('---eos-k', type=float, default=1.5)
parser.add_argument('--vv_thr', type=float, default=0.05)
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

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

setup_seeds()
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"


# ========================================
#             Model Initialization
# ========================================


print('Initializing Model')

config =  OmegaConf.load(args.cfg_path)
model_config = build_model_config(config)
model_config.device_8bit = args.gpu_id

model = LLaVa.from_config(model_config['model']).to(device)
model.eval()

preprocess_config = build_preprocessor_config()

vis_processors = dict()
vis_proc_cfg = preprocess_config.get("vis_processor")
vis_eval_cfg = vis_proc_cfg.get("eval")['proc_type']
vis_processors["eval"] = ClipImageEvalProcessor(vis_eval_cfg)

# vis_processors.do_normalize = False
print(vis_processors["eval"].transform)
print("Done!")




mean = (0.48145466, 0.4578275, 0.40821073)
std = (0.26862954, 0.26130258, 0.27577711)
norm = transforms.Normalize(mean, std)


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
    image = vis_processors["eval"](raw_image).unsqueeze(0)
    image = image.to(device)

    qu = "Please describe this image in detail."
    template = "USER: <ImageHere> <question> ASSISTANT:"
    qu = template.replace("<question>", qu)
    


    with torch.inference_mode():
        with torch.no_grad():
            out = model.generate(
                prompt = qu,
                image = image.half(),
                use_nucleus_sampling=args.sample, # True,
                max_new_tokens=512,
                use_cache=True,
                sample_greedy = args.sample_greedy, # true
                num_beams=1,
            )
    img_save["caption"] = out[0]
    with open(save_path, "a") as f:
                json.dump(img_save, f)
                f.write('\n')
               
                
                