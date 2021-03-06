# -*- coding: utf-8 -*-
import shutil
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('last_model')
# model = SentenceTransformer('cl-tohoku/bert-base-japanese-char-whole-word-masking')

import torch
import tqdm

#Compute embedding for both lists
sentences1 = []
with open("jap1.txt", 'r') as file:
    for line in file:
        line.replace("\n",'')
        sentences1.append(line)

sentences2 = []
with open("jap2.txt", 'r') as file:
    for line in file:
        line.replace("\n",'')
        sentences2.append(line)

import torch
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
model.to(device)

embeddings1 = model.encode(sentences1, convert_to_tensor=True)
embeddings2 = model.encode(sentences2, convert_to_tensor=True)

#Compute cosine-similarits
cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)

shutil.rmtree('result.txt', ignore_errors=True)
with open('result.txt','a+') as f:
    for i in tqdm.tqdm(range(len(sentences1))):
        f.write(str(float(max(0,cosine_scores[i][i]*10)))+"\n")
