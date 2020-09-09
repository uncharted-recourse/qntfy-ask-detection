#
# GRPC Server for Qntfy Ask Detection Service
# 
# Uses GRPC service config in protos/grapevine.proto
# 

import os
import sys
import numpy as np
import pickle
import tensorflow as tf
from concurrent import futures
import time
import logging
import grpc
import json
from flask import Flask


# Keras utils
from ask_detector import *

import grapevine_pb2
import grapevine_pb2_grpc


restapp = Flask(__name__)


# GLOBALS
GRPC_PORT = '50051'

_ONE_DAY_IN_SECONDS = 60 * 60 * 24

BASE_DIR = os.getcwd()
print('Current working directory:', BASE_DIR)

MODEL_PATH = os.path.join(BASE_DIR, 'models/cnn_classifier.h5')
print(MODEL_PATH)

VOCAB_PATH = os.path.join(BASE_DIR, 'data/swda_vocab.txt')
print(VOCAB_PATH)

LABEL_PATH = os.path.join(BASE_DIR, 'data/labels.json')

#-----
class AskDetectionClassifier(grapevine_pb2_grpc.ClassifierServicer):

    # Main classify function
    def Classify(self, request, context):

        # init classifier result object
        result = grapevine_pb2.Classification(
            domain='ask',
            prediction='false',
            confidence=0.0,
            model="qntfy-ask-detection",
            version="0.0.1",
            meta=grapevine_pb2.Meta(),
        )

        # global graph
        with graph.as_default():
            # get text from input message
            input_doc = request.text
            print('input_doc:')
            print(input_doc)
            
            # Exception cases
            if (len(input_doc.strip()) == 0) or (input_doc is None):
                return result

            prediction_data = MODEL_OBJECT.predict(input_doc=input_doc)
            print('prediction_data:')
            print(prediction_data)

            if 'sentences' in prediction_data:
                clf_metadata, any_ask = self.format_metadata(prediction_data['sentences'])
                result.meta.CopyFrom(clf_metadata)
            if any_ask:
                result.prediction = 'true'

            print(prediction_data)

            return result


    # Convert classifier 'sentences' metadata into required protobuf format
    def format_metadata(self, sentences):
        ask_labels = {'open_question', 'yes_no_question', 'action_directive', 'wh_question'}
        any_ask = False
        thisMeta = grapevine_pb2.Meta()
        askList = []
        for s in sentences:
            thisAsk = grapevine_pb2.Ask()
            thisAsk.sentence = s['sentence']
            thisAsk.raw_sentence = s['sentence']
            thisAsk.prediction = s['prediction']
            # HACK - check if predicted class is in `ask_labels`
            if s['prediction'] in ask_labels:
                any_ask = True
            thisAsk.confidence = s['confidence']
            thisAsk.start_idx = s['start_idx']
            thisAsk.end_idx = s['end_idx']

            askList.append(thisAsk)

        thisMeta.asks.extend(askList)

        return thisMeta, any_ask


#-----
def load_model_obj():
    global graph
    global MODEL_OBJECT

    MODEL_OBJECT = AskDetector(model_path=MODEL_PATH, vocab_path=VOCAB_PATH, label_path=LABEL_PATH)
    print('Model loaded successfully!')
    graph = tf.get_default_graph()  


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    grapevine_pb2_grpc.add_ClassifierServicer_to_server(AskDetectionClassifier(), server)
    server.add_insecure_port('[::]:' + GRPC_PORT)
    server.start()
    restapp.run()

    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)

@restapp.route("/healthcheck")
def health():
    return "HEALTHY"


if __name__ == '__main__':
    logging.basicConfig()
    load_model_obj()        # Load model
    serve()
    