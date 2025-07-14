
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
from datetime import datetime




from utils.pope_loader_llava import POPEDataSet


from utils.evrb_llava_sample import evolve_my_sampling
from utils.evrb_llava import LLaVa
from omegaconf import OmegaConf
from utils.hyper_config import hyper_param

from pathlib import Path
import warnings






warnings.filterwarnings("ignore")
time = datetime.now().strftime('%m-%d-%H:%M')
print(time)




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
    parser = argparse.ArgumentParser(description="POPE evaluation on LVLMs.")
    parser.add_argument("--gpu-id", type=str,  default='0', help="specify the GPUs to load the model.")
    parser.add_argument("--data-path", type=str, default="../datasets/coco/val2014", help="data path")
    parser.add_argument("--gqa-data-path", type=str, default="../datasets/gqa/images", help="data path")
    parser.add_argument("--cfg-path", type=str, default="./utils/llava/config/llava-1.5_eval.yaml", help="cfg path")
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

    hyper_param.img_ent_thr = args.img_ent_thr
    hyper_param.pri_rec_thr = args.pri_rec_thr
    hyper_param.do_ct = args.do_ct
    hyper_param.do_eos = args.do_eos

    if args.do_ct:
        evolve_my_sampling()

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



def main():

    args = parse_args()

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
    # load pope data

    for pope_name, pope_path in POPE_PATH.items():
        if 'gpa'in pope_name:
            data_path = args.gqa_data_path
        else:
            data_path = args.data_path

        pope_dataset = POPEDataSet(
            pope_path= pope_path, 
            data_path=data_path, 
            trans=vis_processors["eval"]
        )
        pope_loader = torch.utils.data.DataLoader(
            pope_dataset, 
            batch_size=args.batch_size, 
            shuffle=False, 
            num_workers=args.num_workers,
            drop_last=False
        )

        print ("load data finished")


        print("Start eval...")
        pred_list, label_list = [], []
        
        bar = pope_loader
        # bar = pope_loader
        for idx, data in enumerate(bar): 
            image = data["image"]
            qu = data["query"]
            label = data["label"]
            label_list = label_list + list(label)

            template = "USER: <ImageHere> <question> ASSISTANT:"
            qu = [template.replace("<question>", q) for q in qu]

            image = image.to(device)
            label = torch.Tensor(label).to(device)

            with torch.inference_mode():
                with torch.no_grad():
                    out = model.generate(
                        prompt = qu,
                        image = image.half(),
                        use_nucleus_sampling=args.sample, # True,
                        max_new_tokens=1,
                        use_cache=True,
                        sample_greedy = args.sample_greedy, # true
                        num_beams=1,
                    )
                    pred_list = recorder(out, pred_list) # 'No' pred_list append 0, else 1

        print("===============================================")
        print(pope_name)
        print_acc(pred_list, label_list)








if __name__ == "__main__":
    main()
