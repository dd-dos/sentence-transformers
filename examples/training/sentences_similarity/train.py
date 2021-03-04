"""
This examples trains BERT (or any other transformer model like RoBERTa, DistilBERT etc.) for the STSbenchmark from scratch. It generates sentence embeddings
that can be compared using cosine-similarity to measure the similarity.

Usage:
python training_nli.py

OR
python training_nli.py pretrained_transformer_model_name
"""
from torch.utils.data import DataLoader
import math
from ....sentence_transformers import SentenceTransformer,  SentencesDataset, LoggingHandler, losses, models, util
from ....sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from ....sentence_transformers.readers import STSBenchmarkDataReader, InputExample
import logging
from datetime import datetime
import sys
import os
import gzip
import csv
import argparse
import numpy as np

def main(args):
    #### Just some code to print debug information to stdout
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])
    #### /print debug information to stdout


    #You can specify any huggingface/transformers pre-trained model here, for example, bert-base-uncased, roberta-base, xlm-roberta-base
    model_name = 'cl-tohoku/bert-base-japanese-char-whole-word-masking'

    # Read the dataset
    train_batch_size = 64
    num_epochs = 1000
    model_save_path = os.path.join(args.save_path, model_name.replace("/", "-")+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    # Use Huggingface/transformers model (like BERT, RoBERTa, XLNet, XLM-R) for mapping tokens to embeddings
    word_embedding_model = models.Transformer(model_name)

    # Apply mean pooling to get one fixed sized sentence vector
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                pooling_mode_mean_tokens=True,
                                pooling_mode_cls_token=False,
                                pooling_mode_max_tokens=False)

    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    # Convert the dataset to a DataLoader ready for training
    logging.info("Read custom train dataset")

    train_samples = []
    val_samples = []
    inp_list = []
    dataset_path = args.data_path
    with gzip.open(dataset_path, 'rt', encoding='utf8') as fIn:
        reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
        for row in reader:
            score = float(row['score']) / 10  # Normalize score to range 0 ... 1
            inp_list.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=score))


    from sklearn.model_selection import train_test_split
    train_samples, val_samples = train_test_split(inp_list, test_size=0.2)
    # import ipdb; ipdb.set_trace()


    train_dataset = SentencesDataset(train_samples, model)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
    train_loss = losses.CosineSimilarityLoss(model=model)


    logging.info("Read custom dev dataset")
    # evaluator = EmbeddingSimilarityEvaluator.from_input_examples(val_samples, name='sts-dev')
    evaluator = EmbeddingSimilarityEvaluator.from_input_examples(val_samples)


    # Configure the training. We skip evaluation in this example
    warmup_steps = math.ceil(len(train_dataset) * num_epochs / train_batch_size * 0.1) #10% of train data for warm-up
    logging.info("Warmup-steps: {}".format(warmup_steps))

    # import ipdb; ipdb.set_trace()
    # Train the model
    model.fit(train_objectives=[(train_dataloader, train_loss)],
            evaluator=evaluator,
            epochs=num_epochs,
            evaluation_steps=1000,
            warmup_steps=warmup_steps,
            output_path=model_save_path)

def argparser():
    P = argparse.ArgumentParser(description='training arguments')
    P.add_argument('--data_path', type=str, required=True, help='path to dataset')
    P.add_argument('--save_path', type=str, required=True, help='model save path')

    args = P.parse_args()
    return args

if __name__=="__main__":
    args = argparser()
    main(args)
