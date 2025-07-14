import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from datetime import datetime



from utils.pope_loader_evrb import POPEDataSet


import logging
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
time = datetime.now().strftime('%m-%d-%H:%M')
print(time)


from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


from utils.evrb_qwen_sample import evolve_my_sampling   

from utils.hyper_config import hyper_param




POPE_PATH = {
    "coco_random": "pope/coco/coco_pope_random.json",
    "coco_popular": "pope/coco/coco_pope_popular.json",
    "coco_adversarial": "pope/coco/coco_pope_adversarial.json",
    "gpa_random": "pope/gpa/gqa_pope_seem_random.json",
    "gpa_popular": "pope/gpa/gqa_pope_seem_popular.json",
    "gpa_adversarial": "pope/gpa/gqa_pope_seem_adversarial.json",
    "aokvqa_random": "pope/aokvqa/aokvqa_pope_seem_random.json",
    "aokvqa_popular": "pope/aokvqa/aokvqa_pope_seem_popular.json",
    "aokvqa_adversarial": "pope/aokvqa/aokvqa_pope_seem_adversarial.json",
}



def parse_args():
    parser = argparse.ArgumentParser(description="POPE-Adv evaluation on LVLMs.")
    parser.add_argument("--gpu-id", type=str,  default='0', help="specify the GPUs to load the model.")
    parser.add_argument("--data-path", type=str, default="../datasets/coco/val2014", help="data path")
    parser.add_argument("--ckpt-path", type=str, default="../Qwen2.5-VL/Qwen/Qwen2.5-VL-7B-Instruct", help="ckpt path")
    parser.add_argument("--batch-size", type=int, default=1, help="batch size")
    parser.add_argument("--num_workers", type=int, default=1, help="num workers")
    parser.add_argument("--answers-file", type=str, default="")

    parser.add_argument("--sample-greedy", action='store_true',  default=True)
    parser.add_argument("--sample", action='store_true',  default=True)

    parser.add_argument("--options")


    # my args
    parser.add_argument("--img-ent-thr", type=float, default=7.48)
    parser.add_argument("--pri-rec-thr", type=float, default=0.1)
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
    return args






def setup_seeds():
    seed = 42

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def print_acc(pred_list, label_list):
    pos = 1
    neg = 0
    yes_ratio = pred_list.count(1) / len(pred_list)
    # unknown_ratio = pred_list.count(2) / len(pred_list)

    TP, TN, FP, FN = 0, 0, 0, 0
    for pred, label in zip(pred_list, label_list):
        if pred == pos and label == pos:
            TP += 1
        elif pred == pos and label == neg:
            FP += 1
        elif pred == neg and label == neg:
            TN += 1
        elif pred == neg and label == pos:
            FN += 1

    print('TP\tFP\tTN\tFN\t')
    print('{}\t{}\t{}\t{}'.format(TP, FP, TN, FN))

    precision = float(TP) / float(TP + FP)
    recall = float(TP) / float(TP + FN)
    f1 = 2*precision*recall / (precision + recall)
    acc = (TP + TN) / (TP + TN + FP + FN)
    print('Accuracy: {}'.format(acc))
    print('Precision: {}'.format(precision))
    print('Recall: {}'.format(recall))
    print('F1 score: {}'.format(f1))
    print('Yes ratio: {}'.format(yes_ratio))


def recorder(out, pred_list):
    NEG_WORDS = ["No", "not", "no", "NO"]
    for line in out:

        line = line.replace('.', '')
        line = line.replace(',', '')
        words = line.split(' ')
        if any(word in NEG_WORDS for word in words) or any(word.endswith("n't") for word in words):
            pred_list.append(0)
        else:
            pred_list.append(1)
    
    return pred_list


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

entropy_st =  [0 for i in range(18)]
def statistic(dict):
    global entropy_st
    value = dict['value']
    count = dict['count']
    for i,j in zip(value, count):
        entropy_st[i] += j
    return 


def main():

    args = parse_args()


    # print(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    setup_seeds()
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    # ========================================
    #             Model Initialization
    # ========================================
    print('Initializing Model')


    ckpt_path = args.ckpt_path
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





    # vis_processors, txt_processors = load_preprocess(cfg.get_config().preprocess)
    # # vis_processors.do_normalize = False
    # print(vis_processors["eval"].transform)
    # print("Done!")

    # 获取所有的数据
    data_loaders = []
    for dp in POPE_PATH.values():
        if 'gqa'in dp:
            data_path = args.gqa_data_path
        else:
            data_path = args.data_path
        # load pope data
        pope_dataset = POPEDataSet(
            pope_path=dp, 
            data_path=data_path, 
        )
        data_loaders.append(torch.utils.data.DataLoader(
            pope_dataset, 
            batch_size=args.batch_size, 
            shuffle=False, 
            num_workers=args.num_workers,
            drop_last=False
        ) )     
    data_names = [n for n in POPE_PATH.keys()]

    print ("load data finished")
    
    # hyper_param = {
    # 'img_ent_thr' : args.img_ent_thr,
    # 'pri_rec_thr' : args.pri_rec_thr
    # }

    for idx, pope_loader in enumerate(data_loaders):
        print("Start eval...")        
        pred_list, pred_list_s, label_list = [], [], []
        # for batch_id, data in tqdm(enumerate(pope_loader), total=len(pope_loader)):

        for batch_id, data in enumerate(pope_loader):
            image = data["image_path"]
            qu = data["query"]
            label = data["label"]
            label_list = label_list + list(label)
            messages[0]['content'][0]["image"] = image[0]
            messages[0]['content'][1]["text"] = qu[0]
            
            
            # #### 
            # if old_img_path == image[0]:
            #     continue
            # else:
            #     old_img_path = image[0]
            # ####

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

            # statistic(generated_ids)
            # continue

            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )           
            
            pred_list = recorder(output_text, pred_list)

            # import pdb; pdb.set_trace()
        # # 5638 13401 21879 29045 36276 36106 29950 4129 0 0 0 0
        
        print("===============================================")
        print(data_names[idx])
        if len(pred_list) != 0:
            print_acc(pred_list, label_list)
            
        if len(pred_list_s) != 0:
            print_acc(pred_list_s, label_list)








if __name__ == "__main__":
    main()
