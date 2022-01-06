# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from paddle_serving_server.web_service import WebService, Op
from paddlenlp.data import Pad, Dict
from paddlenlp.transformers import ConvBertTokenizer
from scipy.special import softmax


class ConvBERTOp(Op):
    """ConvBERTOp

    ConvBERT service op
    """

    def init_op(self):
        self.tokenizer = ConvBertTokenizer.from_pretrained("convbert-base")
        self.batchify_fn = lambda samples, fn=Dict({
            # input
            "input_ids": Pad(axis=0, pad_val=self.tokenizer.pad_token_id, dtype="int64"),
            "token_type_ids": Pad(axis=0, pad_val=self.tokenizer.pad_token_type_id,
                                  dtype="int64"),  # segment
        }): fn(samples)

    def preprocess(self, input_dicts, data_id, log_id):
        """preprocess

        In preprocess stage, assembling data for process stage. users can 
        override this function for model feed features.

        Args:
            input_dicts: input data to be preprocessed
            data_id: inner unique id, increase auto
            log_id: global unique id for RTT, 0 default

        Return:
            output_data: data for process stage
            is_skip_process: skip process stage or not, False default
            prod_errcode: None default, otherwise, product errores occured.
                          It is handled in the same way as exception. 
            prod_errinfo: "" default
        """
        (_, input_dict), = input_dicts.items()
        all_text = []
        for key in input_dict.keys():
            text = input_dict[key]
            all_text.append(text)
        data = self.tokenizer(all_text, max_seq_len=128)
        input_ids, token_type_ids = self.batchify_fn(data)
        return {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids
        }, False, None, ""

    def postprocess(self, input_dicts, fetch_dict, data_id, log_id):
        """postprocess

        In postprocess stage, assemble data for next op or output.
        Args:
            input_data: data returned in preprocess stage, dict(for single predict) or list(for batch predict)
            fetch_data: data returned in process stage, dict(for single predict) or list(for batch predict)
            data_id: inner unique id, increase auto
            log_id: logid, 0 default

        Returns: 
            fetch_dict: fetch result must be dict type.
            prod_errcode: None default, otherwise, product errores occured.
                          It is handled in the same way as exception.
            prod_errinfo: "" default
        """
        logits_list = list(fetch_dict.values())[0]
        result = {"label_id": [], "prob": []}
        for logits in logits_list:
            score = softmax(logits, axis=-1)
            label_id = score.argmax()
            prob = score[label_id]
            result["label_id"].append(label_id)
            result["prob"].append(prob)
        result["label_id"] = str(result["label_id"])
        result["prob"] = str(result["prob"])
        return result, None, ""


class ConvBERTService(WebService):
    """ConvBERTService

    ConvBERT service class.
    """

    def get_pipeline_response(self, read_op):
        convbert_op = ConvBERTOp(name="convbert", input_ops=[read_op])
        return convbert_op


# define the service class
uci_service = ConvBERTService(name="convbert")
# load config and prepare the service
uci_service.prepare_pipeline_config("config.yml")
# start the service
uci_service.run_service()
