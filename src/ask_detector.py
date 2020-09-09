import json
import os
import re
import numpy as np

from nltk.tokenize import word_tokenize
from nltk.tokenize.punkt import PunktSentenceTokenizer
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

class TextHandler(object):
    def __init__(self, vocab_path:str, max_len:int):
        self.vocab_path = vocab_path
        self.max_len = max_len
        self._load_vocab()
        self.sent_tokenizer = PunktSentenceTokenizer()

    def _load_vocab(self):
        vocab_map = {}
        with open(self.vocab_path, encoding='utf8', mode='r') as infile:
            for line_ix, line in enumerate(infile):
                word, _ = line.strip().split('\t')
                vocab_map[word] = line_ix + 1

        vocab_map['<PAD>'] = 0
        vocab_map['<UNK>'] = max(vocab_map.values()) + 1

        self.vocab = vocab_map
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

    def _pad_seqs(self, input_seqs:list):
        max_len = max([len(s) for s in input_seqs])
        padded_seqs = pad_sequences(input_seqs, maxlen=max_len, padding='post')

        return padded_seqs

    def get_tok_id(self, w:str):
        if w in self.vocab.keys():
            w_id = self.vocab[w]
        else:
            w_id = self.vocab['<UNK>']

        return w_id

    def process_sent(self, input_sent:str):
        word_toks = word_tokenize(input_sent)
        tok_ids_ = [self.get_tok_id(w) for w in word_toks]
        print()
        print('tok ids:')
        print(tok_ids_)
        print('reverse tok ids:')
        print([self.inverse_vocab[w_id] for w_id in tok_ids_])
        print()
        tok_ids = tok_ids_[:self.max_len]

        return tok_ids

    def process_doc(self, input_doc:str):
        sent_data = []
        for start_idx, end_idx in self.sent_tokenizer.span_tokenize(input_doc.strip()):
            sent_map = {}
            sent = input_doc.strip()[start_idx: end_idx]
            sent_map['sentence'] = sent
            sent_map['start_idx'] = start_idx
            sent_map['end_idx'] = end_idx
            sent_data.append(sent_map)

        sent_toks = [self.process_sent(s['sentence']) for s in sent_data]
        sents_out = self._pad_seqs(sent_toks)

        return sents_out, sent_data


class AskDetector(object):
    def __init__(self, model_path:str, vocab_path:str, label_path:str):
        self.model_path = model_path
        self.label_path = label_path
        self.text_handler = TextHandler(vocab_path=vocab_path, max_len=35)
        self._load_labels()
        self._load_model()

    def _load_labels(self):
        with open(self.label_path, mode='r') as infile:
            label_map = json.load(infile)
        
        label_map = {int(k): v for k, v in label_map.items()}
        self.label_map = label_map

    def _load_model(self):
        self.model = load_model(self.model_path)

    def predict(self, input_doc:str):
        # Store data in format of TA1 payload
        result_map = {}
        result_map['doc_prediction'] = "None"
        result_map['doc_score'] = 0.0

        x_input, sentence_metadata = self.text_handler.process_doc(input_doc)
        print('x_input shape:', x_input.shape)
        y_scores = self.model.predict(x_input)
        y_hat = np.argmax(y_scores, axis=1)

        predicted_classes = [self.label_map[c] for c in y_hat]
        print('Predicted classes:')
        print(predicted_classes)

        class_scores = [y_scores[c_ix][c] for c_ix, c in enumerate(y_hat)]
        output = [{'sentence': sent['sentence'],
                   'start_idx': sent['start_idx'],
                   'end_idx': sent['end_idx'],
                   'prediction': pred,
                   'confidence': score} for sent, pred, score in zip(sentence_metadata, predicted_classes, class_scores)]
        
        result_map['sentences'] = output

        return result_map

if __name__ == '__main__':
    from pprint import pprint

    model_path = '../models/cnn_classifier.h5'
    vocab_path = '../data/swda_vocab.txt'
    label_path = '../data/labels.json'

    ask_detector = AskDetector(model_path, vocab_path, label_path)
    print('Model loaded...')

    dummy_sents = "hey, can you send that again? Yes, I totally agree with you. What was that?"
    output = ask_detector.predict(dummy_sents)

    print('\nReturned output:')
    pprint(output)

