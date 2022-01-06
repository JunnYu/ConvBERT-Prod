import paddle
import paddle.nn.functional as F
import numpy as np
import argparse
from reprod_log import ReprodLogger

from paddlenlp.transformers import ConvBertForSequenceClassification, ConvBertTokenizer


def get_args(add_help=True):
    parser = argparse.ArgumentParser(
        description='PaddleNLP Classification Predict', add_help=add_help)

    parser.add_argument(
        "--text",
        default="the problem , it is with most of these things , is the script .",
        type=str,
        help="SST-2 Text")
    parser.add_argument(
        "--model_path",
        default=None,
        type=str,
        required=True,
        help="Path of the trained model to be exported.", )
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        choices=["gpu", "cpu"],
        help="device", )
    args = parser.parse_args()
    return args


@paddle.no_grad()
def main(args):
    paddle.set_device(args.device)
    # define model
    model = ConvBertForSequenceClassification.from_pretrained(args.model_path)
    tokenizer = ConvBertTokenizer.from_pretrained(args.model_path)
    model.eval()

    tokenized_data = tokenizer(args.text)
    inputs = {
        "input_ids": paddle.to_tensor(
            [tokenized_data["input_ids"]], dtype="int64"),
        "token_type_ids": paddle.to_tensor(
            [tokenized_data["token_type_ids"]], dtype="int64"),
    }

    logits = model(**inputs)
    probs = F.softmax(logits, axis=-1).numpy()[0]
    label_id = probs.argmax()
    prob = probs[label_id]
    print(f"label_id: {label_id}, prob: {prob}")
    return label_id, prob


if __name__ == "__main__":
    args = get_args()
    label_id, prob = main(args)
    reprod_logger = ReprodLogger()
    reprod_logger.add("label_id", np.array([label_id]))
    reprod_logger.add("prob", np.array([prob]))
    reprod_logger.save("output_predict_engine.npy")
