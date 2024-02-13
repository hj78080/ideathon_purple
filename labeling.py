import pandas as pd

def labeling(text):
    texts = text.split('.')
    print("문장 수 : ", len(texts))
    texts_cp = texts.copy()
    labels = []

    for i, t in enumerate(texts_cp) :
        print(f"{i}. {t} : [f/a/x]")
        label = input()

        if label == 'f' : labels.append(0)
        elif label == 'x' : texts.remove(t)
        else : labels.append(1)

    return texts, labels


def write_excel(text):
    df = pd.read_excel('./dataset/labeled_dataset.xlsx')

    texts, labels = labeling(text)

    # 새로운 데이터프레임 생성
    new_data = pd.DataFrame({'Text': texts, 'Label': labels})

    # 기존 데이터프레임과 새로운 데이터프레임을 이어붙이기
    df = pd.concat([df, new_data], ignore_index=True)

    # 데이터프레임을 엑셀 파일에 저장
    df.to_excel('./dataset/labeled_dataset.xlsx', index=False)


def read_excel():
    df = pd.read_excel('./dataset/labeled_dataset.xlsx')

    texts = df['Text'].tolist()
    labels = df['Label'].tolist()

    return texts, labels