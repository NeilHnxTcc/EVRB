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

from PIL import Image

from transformers import LlavaForConditionalGeneration, AutoProcessor


from utils.evrb_sample import evolve_my_sampling
evolve_my_sampling()
from utils.hyper_config import hyper_param



POPE_PATH = {
    "426": "pope/coco/426.json",
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

pope_path = {
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

INSTRUCTION_TEMPLATE = {
    "minigpt4": "###Human: <Img><ImageHere></Img> <question> ###Assistant:",
    "instructblip": "<ImageHere><question>",
    "lrv_instruct": "###Human: <Img><ImageHere></Img> <question> ###Assistant:",
    "shikra": "USER: <im_start><ImageHere><im_end> <question> ASSISTANT:",
    "llava-1.5": "USER: <ImageHere> <question> ASSISTANT:"
}


def parse_args():
    parser = argparse.ArgumentParser(description="POPE-Adv evaluation on LVLMs.")
    parser.add_argument("--model", type=str, default="llava-1.5", help="model")
    parser.add_argument("--pope-type", type=str, default="coco_adversarial", help="model")
    parser.add_argument("--gpu-id", type=int, default=2, help="specify the gpu to load the model.")
    parser.add_argument("--data-path", type=str, default="/mnt/gemininjceph2/geminicephfs/wx-mm-spr-xxxx/neilnxhu/datasets/coco/val2014", help="data path")
    parser.add_argument("--gqa-data-path", type=str, default="/mnt/gemininjceph2/geminicephfs/wx-mm-spr-xxxx/neilnxhu/datasets/gqa/images", help="data path")
    parser.add_argument("--batch-size", type=int, default=1, help="batch size")
    parser.add_argument("--num_workers", type=int, default=1, help="num workers")
    parser.add_argument("--options")

    # my args
    parser.add_argument("--img-ent-thr", type=float, default=7.48)
    parser.add_argument("--pri-rec-thr", type=float, default=0.1)
    parser.add_argument("--do-ct", action='store_true', default=False)
    parser.add_argument("--do-eos", action='store_true', default=False)
    args = parser.parse_args()

    hyper_param.img_ent_thr = args.img_ent_thr
    hyper_param.pri_rec_thr = args.pri_rec_thr
    hyper_param.do_ct = args.do_ct
    hyper_param.img_id = 32000
    hyper_param.stop_id = 29889

    return args




def setup_seeds():
    seed = 42

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def print_acc(pred_list, label_list, logger):
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
    logger.info(('Accuracy: {}'.format(acc)))
    logger.info(('Precision: {}'.format(precision)))
    logger.info(('Recall: {}'.format(recall)))
    logger.info(('F1 score: {}'.format(f1)))
    logger.info(('Yes ratio: {}'.format(yes_ratio)))


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
    def log_string(str):
        logger.info(str)
        print(str)
    exp_dir = Path(os.path.join('results', 'log'))
    exp_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath(args.pope_type)
    log_dir.mkdir(exist_ok=True)
    logger = logging.getLogger(args.model)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/log.txt' % log_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)


    # print(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    setup_seeds()
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    # ========================================
    #             Model Initialization
    # ========================================
    print('Initializing Model')


    ckpt_path = "/mnt/gemininjceph2/geminicephfs/wx-mm-spr-xxxx/neilnxhu/EVRB_github/ckpts/llava-v1.5-7b-hf"
 
    model = LlavaForConditionalGeneration.from_pretrained(
            ckpt_path, torch_dtype="auto", device_map=device, attn_implementation='eager',
    )  

    model.eval()

    processor = AutoProcessor.from_pretrained(ckpt_path)





    # vis_processors, txt_processors = load_preprocess(cfg.get_config().preprocess)
    # # vis_processors.do_normalize = False
    # print(vis_processors["eval"].transform)
    # print("Done!")

    # 获取所有的数据
    data_loaders = []
    for dp in pope_path.values():
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
    data_names = [n for n in pope_path.keys()]

    print ("load data finished")
    
    # hyper_param = {
    # 'img_ent_thr' : args.img_ent_thr,
    # 'pri_rec_thr' : args.pri_rec_thr
    # }

    for idx, pope_loader in enumerate(data_loaders):
        print("Start eval...")        
        pred_list, pred_list_s, label_list = [], [], []
        # for batch_id, data in tqdm(enumerate(pope_loader), total=len(pope_loader)):

        for  data in tqdm(pope_loader):
            image_path = data["image_path"][0]
            qu = data["query"][0]
            label = data["label"]
            label_list = label_list + list(label)
            raw_image = Image.open(image_path).convert("RGB")
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

            

            # Inference: Generation of the output
            output = model.generate(**inputs, max_new_tokens=1, output_attentions=args.do_eos)

            start_idx = inputs['input_ids'].shape[-1]
            output_text = processor.decode(output[0][start_idx:], skip_special_tokens=True)
            pred_list = recorder([output_text], pred_list) # 'No' pred_list append 0, else 1
            
        print("[{}, {}]===============================================".format(args.scale_factor, args.num_attn_candidates))
        if len(pred_list) != 0:
            print_acc(pred_list, label_list, logger)
            
        if len(pred_list_s) != 0:
            print_acc(pred_list_s, label_list, logger)








if __name__ == "__main__":
    main()
