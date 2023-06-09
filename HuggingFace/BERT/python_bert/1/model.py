
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
            inp = pb_utils.get_input_tensor_by_name(request, "input_ids")
            attention_mask = pb_utils.get_input_tensor_by_name(request, "attention_mask")
            token_type_ids = pb_utils.get_input_tensor_by_name(request, "token_type_ids")
            input_ids = torch.from_numpy(np.array(inp.as_numpy())).long().to(self.device)
            attention_mask = torch.from_numpy(np.array(attention_mask.as_numpy())).long().to(self.device)
            token_type_ids = torch.from_numpy(np.array(token_type_ids.as_numpy())).long().to(self.device)

            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

            inference_response = pb_utils.InferenceResponse(output_tensors=[
                pb_utils.Tensor(
                    "last_hidden_state",
                    outputs.last_hidden_state.detach().cpu().numpy()
                )
            ])
            responses.append(inference_response)

        return responses

