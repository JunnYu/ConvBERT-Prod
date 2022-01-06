# -*- coding: utf-8 -*-
import paddle.inference as paddle_infer
from paddlenlp.datasets import load_dataset
import json
import paddle
from paddlenlp.data import Stack, Dict, Pad
import paddlenlp


# 根据本地文件格式定义数据读取生成器
def read(filename):
    with open(filename, "r", encoding="utf8") as f:
        data = json.load(f)
    for entry in data:
        yield {
            'words': entry[0],
            'slots': entry[1],
            'intents': entry[2],
            'history': entry[3],
        }


# 1. 创建配置对象，设置预测模型路径
config = paddle_infer.Config("infer_model/model.pdmodel",
                             "infer_model/model.pdiparams")
MODEL_NAME = r'C:\Users\issuser\.paddlenlp\models\ernie-1.0'
tokenizer = paddlenlp.transformers.ErnieTokenizer.from_pretrained(MODEL_NAME)
# 启用 GPU 进行预测 - 初始化 GPU 显存 100M, Deivce_ID 为 0
# config.enable_use_gpu(100, 0)
config.disable_gpu()

# 2. 根据配置内容创建推理引擎
predictor = paddle_infer.create_predictor(config)
# 3. 设置输入数据
# 获取输入句柄
input_handles = [
    predictor.get_input_handle(name) for name in predictor.get_input_names()
]
# 获取输入数据
dev_ds = load_dataset(read, filename='test.json', lazy=False)
dev_batch_sampler = paddle.io.BatchSampler(dev_ds, batch_size=8, shuffle=False)
batchify_fn = lambda samples, fn=Dict({
    "input_ids": Pad(axis=0, pad_val=tokenizer.pad_token_id),
    "token_type_ids": Pad(axis=0, pad_val=tokenizer.pad_token_type_id),
    "slot_labels": Pad(axis=0, pad_val=0, dtype="int64"),
    "intent_labels": Stack(dtype="float32"),
    "history_ids": Pad(axis=0, pad_val=tokenizer.pad_token_id)
}): fn(samples)
dev_data_loader = paddle.io.DataLoader(
    dataset=dev_ds,
    batch_sampler=dev_batch_sampler,
    collate_fn=batchify_fn,
    return_list=True)
data = batchify_fn([dev_ds[0]])

# 设置输入数据
for input_field, input_handle in zip(data, input_handles):
    input_handle.copy_from_cpu(input_field)

# 4. 执行预测
predictor.run()

# 5. 获取预测结果
# 获取输出句柄
output_handles = [
    predictor.get_output_handle(name) for name in predictor.get_output_names()
]
# 从输出句柄获取预测结果
output = [output_handle.copy_to_cpu() for output_handle in output_handles]
# 打印预测结果
print(output)
