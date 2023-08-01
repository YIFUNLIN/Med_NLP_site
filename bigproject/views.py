from django.shortcuts import render
from django.http import HttpResponse
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer
from datasets import Dataset
import numpy as np
import pandas as pd
import gdown
import zipfile

id2label_dict = {
    0: "O",
    1: "B-BODY",
    2: "I-BODY",
    3: "B-SYMP",
    4: "I-SYMP",
    5: "B-INST",
    6: "I-INST",
    7: "B-EXAM",
    8: "I-EXAM",
    9: "B-CHEM",
    10: "I-CHEM",
    11: "B-DISE",
    12: "I-DISE",
    13: "B-DRUG",
    14: "I-DRUG",
    15: "B-SUPP",
    16: "I-SUPP",
    17: "B-TREAT",
    18: "I-TREAT",
    19: "B-TIME",
    20: "I-TIME"
}

# 下载模型
file_id = "1WgqTMCLqzlPApmiS_E87a0k2hESY245q"
output = "model.zip"
gdown.download(
    f"https://drive.google.com/uc?id={file_id}", output, quiet=False)

# 解压模型
with zipfile.ZipFile("model.zip", 'r') as zip_ref:
    zip_ref.extractall("model_directory")  # 替换为你想要的目标目录


def home(request):
    # id2label 字典映射

    tokenizer = AutoTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
    model = AutoModelForTokenClassification.from_pretrained("model_directory")

    user_input = request.POST.get('input_text', '')  # 获取用户的输入

    # 使用模型进行推理
    inputs = tokenizer(user_input, return_tensors="pt")
    outputs = model(**inputs)

    # 将模型的输出转换为要在模板中显示的格式
    predictions = np.argmax(outputs.logits.detach().numpy(), axis=2)[
        0]  # 获取最有可能的类别
    tokens = tokenizer.tokenize(user_input)

    # 个别单词标注
    output_text = ' '.join(
        [f'{token} ({id2label_dict[prediction]})' for token, prediction in zip(tokens, predictions)])

    # 连续标注
    continuous_output = [id2label_dict[prediction]
                         for prediction in predictions]

    # 将用户的输入传递到模板中
    return render(request, 'index.html', {'output_text': output_text, 'continuous_output': continuous_output, 'original_text': user_input})
