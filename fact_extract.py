# 모델 및 옵티마이저 초기화 (예시로 BERT 모델 사용)
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from tqdm import tqdm

# BERT 토크나이저와 모델 로딩
model_name = "kykim/bert-kor-base"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 저장된 모델 불러오기
checkpoint = torch.load('./model/fine_tuned_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])


def classify_fact(text):
    # 텍스트 토큰화 및 패딩
    tokens = tokenizer(text, padding=True, truncation=True, return_tensors="pt")

    # 예측 수행
    with torch.no_grad():
        prediction = model(**tokens)

	# 예측 결과를 바탕으로 감정 출력
    prediction = torch.nn.functional.softmax(prediction.logits, dim=1)
    output = prediction.argmax(dim=1).item()
    labels = ["fact", "argument"]
    return labels[output]


def fact_extract(text):
    text = text.replace("\n", "")
    texts = text.split('.')
    facts = []
    arguments = []

    for t in texts :
        classify_output = classify_fact(t)

        print(t)
        print(classify_output)
        
        if classify_output == 'fact' : facts.append(t)
        else : arguments.append(t)

    return facts, arguments