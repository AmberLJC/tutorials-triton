
import numpy as np
import time
from tritonclient.utils import *
from PIL import Image
import tritonclient.http as httpclient
import requests
import argparse


def main(model_name):
    client = httpclient.InferenceServerClient(url="localhost:8000")

    # Inputs
    text = "What is the fastest car in the world?"
    text_input = np.array([text.encode()], dtype=object).reshape(-1, 1)  # add dtype=object here


    # Set Inputs
    input_tensors = [
        httpclient.InferInput("input_text", text_input.shape, datatype="BYTES")
    ]
    input_tensors[0].set_data_from_numpy(text_input)
    # Set outputs
    outputs = [
        httpclient.InferRequestedOutput("last_hidden_state")
    ]

    print(input_tensors)
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