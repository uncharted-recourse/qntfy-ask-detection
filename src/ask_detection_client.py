#
# Test GRPC client code for Keras Spam Classifier
#
#

from __future__ import print_function
import logging

import grpc

import grapevine_pb2
import grapevine_pb2_grpc

GRPC_PORT = '50051'

def run():

    channel = grpc.insecure_channel('localhost:' + GRPC_PORT)
    stub = grapevine_pb2_grpc.ClassifierStub(channel)

    testMessageQuestions = grapevine_pb2.Message(
        raw="This raw field isn't used...",
        text="Is that a yes or a no? What are you doing right now?",
    )

    testMessageStatements = grapevine_pb2.Message(
        raw="This raw field isn't used...",
        text="That's exactly right! Yes, that's what I was thinking too.",
    )


    ### Test messages that might be classified as questions
    classification = stub.Classify(testMessageQuestions)
    print("Classifier gRPC client received the following document-level asks for Example 1:")
    print(classification)
    print("Classifier gRPC client received the following speech-act detections for Example 1:")
    print(classification.meta)
    print("\n")

    
    ### Test messages that might be classified as statements
    classification = stub.Classify(testMessageStatements)
    print("Classifier gRPC client received the following document-level asks for Example 2:")
    print(classification)
    print("Classifier gRPC client received the following speech-act detections for Example 2:")
    print(classification.meta)
    print("\n")



if __name__ == '__main__':
    logging.basicConfig()
    run()