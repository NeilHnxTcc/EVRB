import argparse
import torch
import os
import json
from tqdm import tqdm
import sys
import os

from PIL import Image
import math
import torch.distributed as dist

from glob import glob




from torchvision import transforms

import random
import numpy as np
import torch.backends.cudnn as cudnn


from PIL import Image

from transformers import LlavaForConditionalGeneration, AutoProcessor


from utils.evrb_llava_sample import evolve_my_sampling
from utils.hyper_config import hyper_param



def setup_seeds():
    seed = 42

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


mean = (0.48145466, 0.4578275, 0.40821073)
std = (0.26862954, 0.26130258, 0.27577711)
norm = transforms.Normalize(mean, std)
eval_type_dict = {
    "Perception": ["existence", "count", "position", "color", "posters", "celebrity", "scene", "landmark", "artwork", "OCR"],
    "Cognition": ["commonsense_reasoning", "numerical_calculation", "text_translation", "code_reasoning"]
}


def recorder(out):
    NEG_WORDS = ["No", "not", "no", "NO"]

    out = out.replace('.', '')
    out = out.replace(',', '')
    words = out.split(' ')
    if any(word in NEG_WORDS for word in words) or any(word.endswith("n't") for word in words):
        return "No"
    else:
        return "Yes"

def load_data(task_name='color', root='./MME/MME_Benchmark_release_version/MME_Benchmark',img_type='jpg'):
    image_folder = os.path.join(root, task_name, 'images')
    qa_folder = os.path.join(root, task_name, 'questions_answers_YN')
    if os.path.exists(image_folder):
        image_files = os.listdir(image_folder)
        qa_files = os.listdir(qa_folder)
        image_files.sort()
        qa_files.sort()
    else:
        path = os.path.join(root, task_name)
        image_folder =path
        qa_folder = path
        files = os.listdir(path)
        image_files, qa_files = [], []
        for file in files:
            if file.endswith(img_type):
                image_files.append(file)
            elif file.endswith('.txt'):
                qa_files.append(file)
        image_files.sort()
        qa_files.sort()
    # import pdb; pdb.set_trace()
    result = {}
    for image_file, qa_file in zip(image_files, qa_files):
        assert image_file.split('.')[0] == qa_file.split('.')[0]
        result[os.path.join(image_folder, image_file)] = open(os.path.join(qa_folder, qa_file), 'r').read()
    return result

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    
    # set up gpu and logging
    
    ##### here we need tokenizer model image_processor 

    ckpt_path = "./ckpts/llava-v1.5-7b-hf"

    model = LlavaForConditionalGeneration.from_pretrained(
        ckpt_path, torch_dtype="auto", device_map=device
    )     
    model.eval()

    processor = AutoProcessor.from_pretrained(ckpt_path)


    

    result_root = os.path.join(args.save_folder, args.save_name)
    os.makedirs(result_root, exist_ok=True)

    for task_name in eval_type_dict:
        if task_name == "Perception":
            img_type = "jpg"
            # continue
        else:
            img_type = "png"
        for eval_type in eval_type_dict[task_name]:
            result_file = os.path.join(result_root, eval_type+'.txt')
            if os.path.exists(result_file):
                os.remove(result_file)
            data = load_data(eval_type, img_type=img_type)
            for image_file, qa_string in data.items():
                # import pdb; pdb.set_trace()
                qa_list = qa_string.split('\n')
                qa_list = qa_list[:2]
                questions = [qa.split('\t')[0] for qa in qa_list]
                gt_answers = [qa.split('\t')[1] for qa in qa_list]
                
                raw_image = Image.open(image_file).convert("RGB")
                for question, gt_answer in zip(questions, gt_answers):
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
                            {"type": "text", "text": question},
                            {"type": "image"},
                            ],
                        },
                    ]    

                    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

                    inputs = processor(images=raw_image, text=prompt, return_tensors='pt').to(0, torch.float16)

                    # Inference: Generation of the output
                    output = model.generate(**inputs, max_new_tokens=1,  output_attentions=args.do_eos)


                    start_idx = inputs['input_ids'].shape[-1]
                    output_text = processor.decode(output[0][start_idx:], skip_special_tokens=True)

                    pred_answer = output_text
                    # Image_Name + "\t" + Question + "\t" + Ground_Truth_Answer + "\t" + Your_Response + "\n"
                    with open(result_file, 'a') as f:
                        line = image_file.split('/')[-1] + "\t" + question + "\t" + gt_answer + "\t" + pred_answer + "\n"
                        # import pdb; pdb.set_trace()
                        f.write(line)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    # vision contrastive decoding

    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    parser.add_argument("--data-path", type=str, default="../data/coco/val2014", help="data path")
    parser.add_argument("--batch-size", type=int, default=1, help="batch size")
    parser.add_argument("--num-workers", type=int, default=1, help="num workers")


    parser.add_argument("--data-folder", type=str, default='./MME/MME_Benchmark_release_version/MME_Benchmark')
    parser.add_argument("--save-folder", type=str, default="./log/results")
    parser.add_argument("--save-name", type=str, default="baseline")

    # my args
    parser.add_argument("--img-ent-thr", type=float, default=7.48)
    parser.add_argument("--pri-rec-thr", type=float, default=0.85)
    parser.add_argument("--do-ct", action='store_true', default=False)
    parser.add_argument("--do-eos", action='store_true', default=False)
    args = parser.parse_args()

    hyper_param.img_ent_thr = args.img_ent_thr
    hyper_param.pri_rec_thr = args.pri_rec_thr
    hyper_param.do_ct = args.do_ct
    hyper_param.img_id = 32000
    hyper_param.stop_id = 29889

    if args.do_ct:
        evolve_my_sampling()

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    setup_seeds()
    eval_model(args)
