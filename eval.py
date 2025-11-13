import os
import re
import json
import argparse
import pandas as pd
from collections import defaultdict
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from minigpt4.common.config import Config
from minigpt4.common.eval_utils import prepare_texts, init_model, eval_parser, computeIoU
from minigpt4.conversation.conversation import CONV_VISION_minigptv2
from minigpt4.common.registry import registry

from minigpt4.datasets.datasets.stance_dataset import stanceDetectionDataset

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def list_of_str(arg):
    return list(map(str, arg.split(',')))

parser = eval_parser()
parser.add_argument("--dataset", type=list_of_str, default='stance_detection_conversation', help="dataset to evaluate")
parser.add_argument("--res", type=float, default=100.0, help="resolution used in refcoco")
parser.add_argument("--resample", action='store_true', help="resolution used in refcoco")
parser.add_argument("--output", default='/home/zbw/project/MMCSD/MLLM-SD/MLLM-SD/result/result.csv', help="output file path")

args = parser.parse_args()

print(args.output)
cfg = Config(args)

model, vis_processor = init_model(args)

model.eval()
CONV_VISION = CONV_VISION_minigptv2
conv_temp = CONV_VISION.copy()
conv_temp.system = ""

text_processor_cfg = cfg.datasets_cfg.stance_detection_conversation.text_processor.train
text_processor = registry.get_processor_class(text_processor_cfg.name).from_config(text_processor_cfg)
vis_processor_cfg = cfg.datasets_cfg.stance_detection_conversation.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

if 'stance_detection_conversation' in args.dataset:
    eval_file_path = cfg.evaluation_datasets_cfg["stance_detection_conversation"]["ann_path"]
    img_path = cfg.evaluation_datasets_cfg["stance_detection_conversation"]["image_path"]
    target = cfg.evaluation_datasets_cfg["stance_detection_conversation"]["target"]
    batch_size = cfg.evaluation_datasets_cfg["stance_detection_conversation"]["batch_size"]
    max_new_tokens = cfg.evaluation_datasets_cfg["stance_detection_conversation"]["max_new_tokens"]
    print(eval_file_path)
    print(img_path)
    print(batch_size)
    print(max_new_tokens)

    data = stanceDetectionDataset(vis_processor, text_processor, img_path, eval_file_path,target)

    eval_dataloader = DataLoader(
            data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=os.cpu_count(),       
            pin_memory=True,                  
            persistent_workers=False         
        )
    minigpt4_predict = []

    names_list = []  
    answers_list = []
    label = []
    with torch.no_grad():
        for batch in tqdm(eval_dataloader):
            images = batch['image'].cuda()
            instruction_input = batch['instruction_input']
            names = batch['name']
            answers = model.generate(images, instruction_input, max_new_tokens=max_new_tokens, do_sample=False)
            answers_list.extend(answers)  
            label.extend(batch['answer'])
            names_list.extend(names)
            del images
            torch.cuda.empty_cache()

    predict = []
    for answer in answers_list:
        if 'against' in answer.lower():
            predict.append('against')
        elif 'favor' in answer.lower():
            predict.append('favor')
        else:
            predict.append('none')

    with open(args.output,'w')as file:
        dic = {}
        dic['label'] = label
        dic['predict'] = predict
        dic['answer'] = answers_list
        pd.DataFrame(dic).to_csv(file)



