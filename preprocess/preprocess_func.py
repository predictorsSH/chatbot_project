import re
import tensorflow_datasets as tfds
import pandas as pd
import os


def data_load(file_path : str ):

    data=pd.read_csv(file_path)
    # questions, answers = list(data['Q']), list(data['A'])
    return data

def make_dataset(data):

    questions = []
    for q in data['Q']:
        q = punctuation_split(q)
        questions.append(q)

    answers = []
    for a in data['A']:
        a = punctuation_split(a)
        answers.append(a)

    return questions, answers


def punctuation_split(sentence : str):

    sentence = re.sub(r'([?.!,])', r' \1 ', sentence)
    sentence = sentence.strip()

    return sentence

def tokenizer_(questions :list , answers :list):

    if not os.path.exists('../vocab/vocab.subwords'):
        print('vocab not exist and make vocab..')

        tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            questions + answers, target_vocab_size=2**13)

        tokenizer.save_to_file('../vocab/vocab')


    print('vocab exist and load vocab!')

    tokenizer = tfds.deprecated.text.SubwordTextEncoder.load_from_file('../vocab/vocab')

    START_TOKEN, END_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]
    VOCAB_SIZE = tokenizer.vocab_size + 2

    return tokenizer, START_TOKEN, END_TOKEN, VOCAB_SIZE

def preprocess_sentence(sentence):
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    sentence = sentence.strip()
    return sentence

