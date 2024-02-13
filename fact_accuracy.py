import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from labeling import read_excel  # labeling.py에는 read_excel 함수가 정의되어 있다고 가정합니다.

# 가상의 데이터셋 클래스 정의 (Fact: 0, Argument: 1)
class FactArgumentDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {'text': self.texts[idx], 'label': self.labels[idx]}

# 가상의 테스트 데이터와 레이블 생성
test_texts, test_labels = read_excel()

# BERT 토크나이저와 모델 로딩
model_name = "kykim/bert-kor-base"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 저장한 학습된 모델의 파라미터 로딩
checkpoint = torch.load('./model/fine_tuned_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# 모델을 평가 모드로 설정
model.eval()

# 정확도 계산을 위한 변수
correct_predictions = 0
total_samples = 0
corr = []

# 테스트 데이터에 대한 정확도 계산
for i, text in enumerate(test_texts):
    tokens = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    label = test_labels[i]

    # 예측 수행
    with torch.no_grad():
        prediction = model(**tokens)

    prediction = torch.nn.functional.softmax(prediction.logits, dim=1)
    output = prediction.argmax(dim=1).item()

    if output == label :
        corr.append("O")
        correct_predictions += 1
    else : corr.append("X")
    total_samples += 1


# 전체 테스트 데이터에 대한 정확도 출력
accuracy = correct_predictions / total_samples
print(corr)
print(f"Accuracy : {accuracy:.4f}")