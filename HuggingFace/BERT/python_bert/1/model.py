
import numpy as np
import triton_python_backend_utils as pb_utils
from transformers import BertTokenizer, BertModel
import torch

class TritonPythonModel:
    def initialize(self, args):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased').to(self.device)

    def execute(self, requests):
        responses = []
        for request in requests:


            inp = pb_utils.get_input_tensor_by_name(request, "input_text")
            input_text = np.squeeze(inp.as_numpy())

            # Use the tokenizer to encode the text
            inputs = self.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {name: tensor.to('cuda') for name, tensor in inputs.items()}

            outputs = self.model(**inputs)

            inference_response = pb_utils.InferenceResponse(output_tensors=[
                pb_utils.Tensor(
                    "last_hidden_state",
                    outputs.last_hidden_state.detach().cpu().numpy()
                )
            ])
            responses.append(inference_response)

        return responses

