# from tritonclient.http import InferenceServerClient
import tritonclient.http as httpclient
from transformers import BertTokenizer
import numpy as np
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def bert_inference(input_string):

    encoding = tokenizer(input_string, max_length=128, padding='max_length', truncation=True, return_tensors='np')
    input_ids_np = encoding['input_ids']
    attention_mask_np = encoding['attention_mask']

    # Set up Triton client
    triton_client = httpclient.InferenceServerClient(url="localhost:8000")

    # Prepare the inputs for the request
    inputs = []
    inputs.append(httpclient.InferInput('input_ids', input_ids_np.shape, "INT64"))
    inputs.append(httpclient.InferInput('attention_mask', attention_mask_np.shape, "INT64"))

    # Set the data for the inputs
    inputs[0].set_data_from_numpy(input_ids_np)
    inputs[1].set_data_from_numpy(attention_mask_np)

    # Perform the inference
    result = triton_client.infer("bert", inputs)

    # Get the output tensors from the response
    output = result.as_numpy('output')

    predictions = torch.argmax(torch.tensor(output), dim=-1)
    print(predictions)

    return output

output = bert_inference("Hello, my dog is cute")

