"""
Work Process
0. bert 기반 모델 팩트 추출 데이터 학습시킴, 정확도 확인 - labeling.py, fact_train.py, fact.accuracy.py
1. 기사 데이터를 positive / negative 로 분류함 - emotion_analysis.py
2. 기사 데이터에서 주장을 제거하고 fact를 추출함. 전체 기사를 '.' 단위로 slice 하여 fact 문자열 리스트로 반환 - fact_extract.py
3. fact 문장, positive / negative 로 fine tuning 후 기사 생성 - LLM.py

Gpt fine tuning 절차
1. OpenAI 가입하여 API 키 받기
2. API 키 환경 변수 설정 (setx OPENAI_API_KEY "your-api-key-here")
3. jsonl 형식의 fine tuning용 파일 작성
4. 아래 코드 실행
"""

from openai import OpenAI

client = OpenAI()


# data.jsonl을 바탕으로 파인튜닝
file = client.files.create(
  file=open("./dataset/finetune_dataset.jsonl", "rb"),
  purpose="fine-tune"
)

client.fine_tuning.jobs.create(
  training_file=file.id,
  model="gpt-3.5-turbo"
)

"""
completion = client.chat.completions.create(
  model="ft:gpt-3.5-turbo-0613:personal::8jDSexWY",
  messages=[
    {"role": "system", "content": "중립적인 기사를 작성하는 기자"},
    {"role": "user", "content": "후쿠시마 오염수 방류에 대해, positive와 negative한 입장이 골고루 들어가게 1000자 정도의 새 기사를 생성해줘"}
  ]
)
print(completion.choices[0].message)
"""