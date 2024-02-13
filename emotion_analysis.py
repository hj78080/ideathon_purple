import torch
from transformers import BertTokenizer, BertForSequenceClassification
import torch.nn.functional as F

model_name = "kykim/bert-kor-base"
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
tokenizer = BertTokenizer.from_pretrained(model_name)


def classify_emotion(text):
    # 텍스트 토큰화 및 패딩
    tokens = tokenizer(text, padding=True, truncation=True, return_tensors="pt")

    # 예측 수행
    with torch.no_grad():
        prediction = model(**tokens)

	# 예측 결과를 바탕으로 감정 출력
    prediction = F.softmax(prediction.logits, dim=1)
    print(prediction)
    output = prediction.argmax(dim=1).item()
    labels = ["positive", "negative"]
    return labels[output]

