from django.shortcuts import render
from transformers import AutoTokenizer, AutoModelForTokenClassification
import numpy as np
import gdown
import zipfile
import os

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

tokenizer = None
model = None


def initialize():
    global tokenizer
    global model

    base_dir = os.path.dirname(os.path.abspath(__file__))  # 獲取當前腳本的絕對路徑
    model_directory = os.path.join(
        base_dir, "bert_token")  # 使用絕對路徑指向bert_token資料夾

    if not os.path.exists(model_directory):
        raise ValueError(f"模型資料夾 {model_directory} 不存在!")

    tokenizer = AutoTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
    model = AutoModelForTokenClassification.from_pretrained(model_directory)


def home(request):
    global tokenizer
    global model

    if tokenizer is None or model is None:
        initialize()

    user_input = request.POST.get('input_text', '')

    inputs = tokenizer(user_input, return_tensors="pt")
    outputs = model(**inputs)

    predictions = np.argmax(outputs.logits.detach().numpy(), axis=2)[0]
    tokens = tokenizer.tokenize(user_input)

    output_text = ' '.join(
        [f'{token} ({id2label_dict[prediction]})' for token, prediction in zip(tokens, predictions)])

    continuous_output = [id2label_dict[prediction]
                         for prediction in predictions]

    return render(request, 'index.html', {'output_text': output_text, 'continuous_output': continuous_output, 'original_text': user_input})
