import os
import json
import pickle
import random
import time
import numpy as np
from PIL import Image
import skimage.io as io
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon, Rectangle
from torch.utils.data import Dataset
import webdataset as wds
import cv2
import re

from minigpt4.datasets.datasets.base_dataset import BaseDataset
from minigpt4.datasets.datasets.caption_datasets import CaptionDataset

    
class stanceDetectionDataset(Dataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_path, target):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        self.vis_root = vis_root

        self.vis_processor = vis_processor
        self.text_processor = text_processor
        self.target = target

        self.ann=[]
    
        with open(ann_path, 'r') as f:
            self.ann = json.load(f)



    def __len__(self):
        return len(self.ann)


    def __getitem__(self, index):
        info = self.ann[index]
        example = {
'Tesla':"""Here is an example.
Post: Sold my Tesla Model 3 today, happy to be done with it.
Comment1: Did you have issues like rattling or phantom braking?
Comment2: That's a feature: quality is a gamble 😂
Answer: Comment2's stance towards Tesla is against.
""",

'Post':"""Here is an example.
Post: Religion always complicates conflicts.
Comment1: Muslims already see Christ as a prophet.
Comment2: No, reducing God to a prophet insults Christians.
Answer: Comment2's stance towards religion is against.
""",

'Bitcoin':"""Here is an example.
Post: Bitcoin is pumping again tonight!
Comment1: 40k soon 🚀
Comment2: It never was and never will be dead.
Answer: Comment2's stance towards Bitcoin is favor.
"""
}

#         example = {
# 'Tesla':"""Here is an example.
# Post:Goodbye Tesla 👋🏼
# After 20 months of owning, glad to get rid of this Tesla Model 3 Long Range..
# AMA about my meh time as an adult buying a depreciating asset that I should’ve sold back during the resale boom of summer 2022 lol.
# Comment1:Did you have an issues with it like rattling at highway speeds or phantom braking? Because I’m thinking of selling my 2023 Camry for the Highland model 3.
# Comment2:Yep. Car rattles when I stop playing music. Front pillar speaker pops and makes weird noises. Little gaps here and there. Phantom braked twice. (I rarely use autopilot).
# Comment3:Like you hear the rattles when you stop playing music? Also are they there during low speeds as well?
# Comment4:Only when there’s no music playing. Highway and city.
# Comment5:That's a feature: When you buy one, you never know the quality you're gonna get 😂
# Answer: Comment5's stance towards Bitcoin is against.
# """,

# 'Post':"""Here is an example.
# Post:Yes. That’s what this conflict needs: more religion
# Comment1:Technically, Muslims already have Christ as he's a major prophet in their religion.
# Comment2:Technical it's an insult to Christians to demote thier God from a God to a human messenger of some one else's God.
# Comment3:But they all share the same god so…
# Comment4:No they don't. How is Allah = Jesus = Yahweh ?Even technically inside each religion each sect has a diffrent God. Even if they share the name their characteristic are diffrent.
# Comment5:..but functionally it all comes from the same traditions and history and shit.  They are all wrong of course but it is all rooted in the same basic shit.
# Answer: Stance: Comment5's stance towards Bitcoin is against.
# """,

# 'Bitcoin':"""Here is an example.
# Post:Guess no sleep for Bitcoin Tonight
# Guess we aren’t sleeping tonight guys, let’s go!
# Comment1:40k soon
# Comment2:I think so too, since we know ATH was $64k+ we know what is possible, only the beginning
# Comment3:I think it’s a bull trap and price will consolidate after the pump at lower price point around 30-33k and find new “support”. It’s definitely gonna turn a few heads though, media gonna write about it and other people will get FOMO. Message is clear though, Bitcoin is not dead.
# Comment4:It never was and never will be.
# Answer: Comment4's stance towards Bitcoin is favor. 
# """,
#         }

        image_file = '{}.jpg'.format(info['imageId'])
        image_path = os.path.join(self.vis_root, image_file)
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)
        target = self.target
        p_v = f"The image above shows what the conversation thread contains for the post.The image content is mainly related to {target}"
        text = info['text']
        lines = text.split('\n')
        last_comment_line = None
        for line in reversed(lines):
            if line.startswith('Comment'):
                last_comment_line = line
                break
        
        if last_comment_line:
            match = re.search(r'Comment(\d+):', last_comment_line)
            if match:
                comment_name = match.group(0).replace(':', '')
            else:
                comment_name = "Post"
        else:
            comment_name = "Post"

        caption = info['caption']
        prompt_T=f"The following is a conversation on social media based on a post. All comments are responses to the content of the post, and each comment replies to the previous one.\nConversation:\n{text}\n There are three stances [favor, against, none]. Choose one of the three stances to express {comment_name}’s stance towards '{target}'."""
        stance = self.text_processor(info['stance'])
        instruction = f'<Img><ImageHere></Img>{p_v}{caption}[stance detection]{prompt_T} {example[target]}'


        return {
            "name": info['imageId'],
            "image": image,
            "instruction_input": instruction,
            'answer': stance,
        }
    
