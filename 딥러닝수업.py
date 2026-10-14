import pandas as pd

# 선행연구 비교 데이터 작성 (강의자료 14페이지 Table 1 참고)
data = {
    "측정 부위": ["손목", "전완", "전완", "이두·삼두", "전완", "손바닥"],
    "인원": [50, 80, 5, 40, 21, 5],
    "채널": [8, 8, 4, 12, 4, 2],
    "동작": [
        "박수치기",
        "스마트폰 잠금해제",
        "손동작 6종",
        "손동작 3종",
        "손 펴기",
        "문손잡이 잡고 돌리기",
    ],
    "특징 추출 + 모델": [
        "GAN + DNN",
        "Siamese CNN",
        "DWT/EWT/EMD + CNN",
        "CQT + CNN",
        "DWT/CWT + CNN",
        "CWT + DenseNet161",
    ],
    "정확도(%)": ["97.94", "92.06", "~95.62", "97.50", "~99.21", "99.00"],
}

# 데이터프레임 생성
df = pd.DataFrame(data)

# 터미널에 표 출력
print("=== 선행연구 비교표 (Table 1) ===")
print(df.to_string(index=False))

# 과제 제출용 CSV 파일로 저장
df.to_csv("related_work_table.csv", index=False, encoding="utf-8-sig")
print("\n표가 'related_work_table.csv' 파일로 저장되었습니다.")