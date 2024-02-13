import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from tqdm import tqdm
from labeling import read_excel

# 가상의 데이터셋 클래스 정의 (Fact: 0, Argument: 1)
class FactArgumentDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {'text': self.texts[idx], 'label': self.labels[idx]}

# 가상의 학습 데이터와 레이블 생성
train_texts, train_labels = read_excel()

# 데이터셋 및 데이터로더 생성
train_dataset = FactArgumentDataset(train_texts, train_labels)
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# BERT 토크나이저와 모델 로딩
model_name = "kykim/bert-kor-base"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 옵티마이저 및 손실 함수 설정
optimizer = AdamW(model.parameters(), lr=2e-5)
criterion = torch.nn.CrossEntropyLoss()

# 학습 수행
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

epochs = 10
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}"):
        inputs = tokenizer(batch['text'], padding=True, truncation=True, return_tensors="pt", max_length=512).to(device)
        labels = torch.tensor(batch['label']).to(device)

        optimizer.zero_grad()
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    average_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch + 1}, Average Loss: {average_loss}")

# 학습된 모델 저장
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, 'fine_tuned_model.pth')
