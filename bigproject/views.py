from django.shortcuts import render
from django.http import HttpResponse
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer
from datasets import Dataset
import numpy as np
import pandas as pd

"""
def home(request):
    user_input = request.GET.get('input_text', '')  # 獲取用戶的輸入
    return render(request, 'index.html', {'output_text': user_input})  # 將用戶的輸入傳遞到模板中
"""

"""
def home(request):
    if request.method == 'POST':
        user_input = request.POST.get('input_text', '')  # 獲取用戶的輸入
        # 將用戶的輸入傳遞到模板中
        return render(request, 'index.html', {'output_text': user_input})
    else:
        return render(request, 'index.html')
"""

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

"""
def home(request):
    if request.method == 'POST':
        tokenizer = AutoTokenizer.from_pretrained(
            'hfl/chinese-roberta-wwm-ext')
        model = AutoModelForTokenClassification.from_pretrained("bert_token")
        trainer = Trainer(model, tokenizer=tokenizer)

        user_input = request.POST.get('input_text', '')  # 獲取用戶的輸入

        # 使用模型進行推理
        inputs = tokenizer(user_input, return_tensors="pt")
        outputs = model(**inputs)

        # 將模型的輸出轉換為要在模板中顯示的格式
        predictions = np.argmax(outputs.logits.detach().numpy(), axis=2)[
            0]  # 獲取最有可能的類別
        tokens = tokenizer.tokenize(user_input)
        output_text = ' '.join(
            [f'{token} ({label_dict[prediction]})' for token, prediction in zip(tokens, predictions)])

        # 將用戶的輸入傳遞到模板中
        return render(request, 'index.html', {'output_text': output_text})
    else:
        return render(request, 'index.html')

"""


def home(request):
    # id2label 字典映射

    tokenizer = AutoTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
    model = AutoModelForTokenClassification.from_pretrained("/bert_token")

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
