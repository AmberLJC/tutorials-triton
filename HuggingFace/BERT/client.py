
import numpy as np
import time
from tritonclient.utils import *
from PIL import Image
import tritonclient.http as httpclient
import requests
import argparse

from transformers import BertTokenizer, BertModel

def main(model_name):
    client = httpclient.InferenceServerClient(url="localhost:8000")

    # Inputs
    input_text = "What is the fastest car in the world?"
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    inputs = tokenizer.encode_plus(input_text, return_tensors='np')

    # Convert the input tensors into the format required by Triton.
    input_tensors = []
    for name, array in inputs.items():
        input_tensors.append(httpclient.InferInput(name, array.shape, np_to_triton_dtype(array.dtype)))
        input_tensors[-1].set_data_from_numpy(array)

    # Set outputs
    outputs = [
        httpclient.InferRequestedOutput("last_hidden_state")
    ]

    # Query
    query_response = client.infer(model_name=model_name,
                                  inputs=input_tensors,
                                  outputs=outputs)

    # Output
    last_hidden_state = query_response.as_numpy("last_hidden_state")
    print(last_hidden_state.shape)
    print(last_hidden_state)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="python_bert")
    args = parser.parse_args()
    main(args.model_name)