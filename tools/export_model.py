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

from paddlenlp.transformers import ConvBertForSequenceClassification, ConvBertTokenizer
import paddle
import os
import argparse


def get_args(add_help=True):
    """get_args

    Parse all args using argparse lib

    Args:
        add_help: Whether to add -h option on args

    Returns:
        An object which contains many parameters used for inference.
    """
    parser = argparse.ArgumentParser(
        description='Paddlenlp Classification Training', add_help=add_help)
    parser.add_argument(
        "--model_path",
        default=None,
        type=str,
        required=True,
        help="Path of the trained model to be exported.", )
    parser.add_argument(
        '--save_inference_dir',
        default='./convbert_infer',
        help='path where to save')

    args = parser.parse_args()
    return args


def export(args):
    # build model
    model = ConvBertForSequenceClassification.from_pretrained(args.model_path)
    tokenizer = ConvBertTokenizer.from_pretrained(args.model_path)
    model.eval()

    # decorate model with jit.save
    model = paddle.jit.to_static(
        model,
        input_spec=[
            paddle.static.InputSpec(
                shape=[None, None], dtype="int64"),  # input_ids
            paddle.static.InputSpec(
                shape=[None, None], dtype="int64")  # token_type_ids
        ])
    # save inference model
    paddle.jit.save(model, os.path.join(args.save_inference_dir, "inference"))
    tokenizer.save_pretrained(args.save_inference_dir)
    print(
        f"inference model and tokenizer have been saved into {args.save_inference_dir}"
    )


if __name__ == "__main__":
    args = get_args()
    export(args)
