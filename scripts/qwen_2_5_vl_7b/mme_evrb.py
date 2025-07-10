import argparse
import torch
import os
from tqdm import tqdm

import os

import math



from torchvision import transforms

import random
import numpy as np
import torch.backends.cudnn as cudnn


from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


from utils.evrb_qwen_sample import evolve_my_sampling   

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

def load_data(task_name='color', root='/mnt/gemininjceph2/geminicephfs/wx-mm-spr-xxxx/neilnxhu/hallucination/MME/MME_Benchmark_release_version/MME_Benchmark',img_type='jpg'):
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
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "/mnt/gemininjceph2/geminicephfs/wx-mm-spr-xxxx/neilnxhu/Qwen2.5-VL/my_example/424_1748933285.mp4",
                },
                {"type": "text", "text": "When did the cat walk away?"},
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
                
                for question, gt_answer in zip(questions, gt_answers):

                    messages[0]['content'][0]["image"] = image_file
                    messages[0]['content'][1]["text"] = question
                    
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
                    generated_ids = model.generate(**inputs, max_new_tokens=10, do_sample=False, output_attentions=args.do_eos)


                    generated_ids_trimmed = [
                        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                    ]
                    output_text = processor.batch_decode(
                        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )           

                    # import pdb; pdb.set_trace()

                    pred_answer = output_text[0]
                    # Image_Name + "\t" + Question + "\t" + Ground_Truth_Answer + "\t" + Your_Response + "\n"
                    with open(result_file, 'a') as f:
                        line = image_file.split('/')[-1] + "\t" + question + "\t" + gt_answer + "\t" + pred_answer + "\n"
                        # import pdb; pdb.set_trace()
                        f.write(line)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    # vision contrastive decoding
    parser.add_argument("--noise-step", type=int, default=500)
    parser.add_argument("--use-cd", action='store_true', default=False)
    parser.add_argument("--use-icd", action='store_true', default=False)
    parser.add_argument("--use-vcd", action='store_true', default=False)
    parser.add_argument("--cd-alpha", type=float, default=1)
    parser.add_argument("--cd-beta", type=float, default=0.1)
    parser.add_argument("--sample-greedy", action='store_true', default=True)
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    parser.add_argument("--data-path", type=str, default="/mnt/gemininjceph2/geminicephfs/wx-mm-spr-xxxx/neilnxhu/data/coco/val2014", help="data path")
    parser.add_argument("--batch-size", type=int, default=1, help="batch size")
    parser.add_argument("--num-workers", type=int, default=1, help="num workers")

    # opera-beamsearch
    parser.add_argument("--test-sample", type=int, default=500)
    parser.add_argument("--beam", type=int, default=1)
    parser.add_argument("--sample", type=bool, default=True)



    parser.add_argument("--data-folder", type=str, default='/mnt/gemininjceph2/geminicephfs/wx-mm-spr-xxxx/neilnxhu/hallucination/MME/MME_Benchmark_release_version/MME_Benchmark')
    parser.add_argument("--save-folder", type=str, default="./log/results")
    parser.add_argument("--save-name", type=str, default="baseline")

    # my args
    parser.add_argument("--img-ent-thr", type=float, default=7.48)
    parser.add_argument("--pri-rec-thr", type=float, default=0.85)
    parser.add_argument("--do-ct", action='store_true', default=False)
    parser.add_argument("--do-eos", action='store_true', default=False)
    args = parser.parse_args()
    
    if args.do_ct:
        evolve_my_sampling()

    hyper_param.img_ent_thr = args.img_ent_thr
    hyper_param.pri_rec_thr = args.pri_rec_thr
    hyper_param.do_ct = args.do_ct
    hyper_param.img_id = 151655
    hyper_param.stop_id = 13

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    setup_seeds()
    eval_model(args)
