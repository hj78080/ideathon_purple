from fact_extract import fact_extract
from emotion_analysis import classify_emotion

# 기사 전체를 받아 감정 분석 -> 팩트 추출 -> 파인튜닝 데이터 생성

news_data = """news"""

emotion = classify_emotion(news_data)
facts, args = fact_extract(news_data)

with open("./dataset/finetune_dataset.jsonl", 'a', encoding='utf-8') as outfile :
    for s in facts :
        w = '{"messages": [{"role": "system", "content": "중립적인 기사를 작성하는 기자"}, {"role": "user", "content": "'+ s +'"}, {"role": "assistant", "content": "'+emotion+'"}]}\n'
        outfile.write(w)