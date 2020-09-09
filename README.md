# qntfy-ask-detection
Simple code and model for making sentence-level speech-act predictions (e.g. statement, question, etc.)
Initial model is a convolutional architecture trained on the SWDA conversation corpus.

## Dockerized gRPC implemenation of classifier

Build docker image by docker-compose in ./docker_clf_project dir:  `docker-compose build`

Run the container using: `docker run -it -p 50051:50051 <docker image id>`
By default, gRPC connection is via port 50051

Test the gRPC server. In a separate command window run the script `python src/spam_clf_client.py`
This sends two separate messages to the server. You should see a set of predicted speech-acts for
each set of messages.
