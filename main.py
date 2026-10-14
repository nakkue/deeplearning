
from pathlib import Path
import re
import pandas as pd
import matplotlib.pyplot as plt

# 한글 그래프 표시
plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 폴더 설정
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "sEMG-Hand-Gesture-Elbow-Angle-Dataset-main" / "data"
OUTPUT_DIR = BASE_DIR / "results"
OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 60)
print("Step2 sEMG 데이터 분석 실습")
print("=" * 60)

if not DATA_DIR.exists():
    raise FileNotFoundError(f"데이터 폴더를 찾을 수 없습니다: {DATA_DIR}")

# 데이터 파일 찾기
files = sorted(DATA_DIR.rglob("*.csv"))
print(f"\nCSV 파일 수: {len(files)}")

if not files:
    raise FileNotFoundError(f"CSV 파일이 없습니다: {DATA_DIR}")

# 파일명에서 각도와 동작 추출
def parse_filename(path):
    name = path.stem.lower()
    angle_match = re.search(r"_(30|60)\b", name)

    if not angle_match:
        return None, None

    angle = int(angle_match.group(1))

    if "fist" in name:
        gesture = "fist"
    elif "open" in name:
        gesture = "open"
    elif "pro" in name:
        gesture = "pro"
    elif "extor" in name:
        gesture = "extor"
    else:
        gesture = None

    return angle, gesture

records = []
for path in files:
    angle, gesture = parse_filename(path)
    records.append({
        "file": path,
        "filename": path.name,
        "angle": angle,
        "gesture": gesture,
        "class": f"{angle}_{gesture}" if angle and gesture else None
    })

summary = pd.DataFrame(records)

print("\n[파일별 클래스 개수]")
print(summary.groupby(["angle", "gesture"], dropna=False).size().to_string())

print("\n[클래스 목록]")
classes = sorted(summary["class"].dropna().unique())
for c in classes:
    print("-", c)

print(f"\n확인된 클래스 수: {len(classes)}")

# 파일 하나 읽기
def read_emg(path):
    df = pd.read_csv(path)

    channels = [
        c for c in df.columns
        if str(c).lower().startswith("bandpass")
    ]

    if not channels:
        raise ValueError(f"EMG 채널을 찾을 수 없습니다: {path.name}")

    for c in channels:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=channels)
    return df, channels

# 데이터셋 구조 요약
print("\n[데이터 구조]")
sample_path = files[0]
sample_df, channel_names = read_emg(sample_path)

print("예시 파일:", sample_path.name)
print("행·열:", sample_df.shape)
print("EMG 채널:", channel_names)
print("채널 수:", len(channel_names))
print("사용 가능한 추가 열:",
      [c for c in sample_df.columns if c not in channel_names])

# 5채널 신호 그래프
plot_path = files[0]
plot_df, channels = read_emg(plot_path)

# 너무 많은 데이터를 한꺼번에 그리지 않도록 앞부분 일부 사용
plot_df = plot_df.iloc[:5000].copy()

if "time" in plot_df.columns:
    x = pd.to_numeric(plot_df["time"], errors="coerce")
    x_label = "Time"
else:
    x = range(len(plot_df))
    x_label = "Sample"

fig, axes = plt.subplots(len(channels), 1, figsize=(12, 9), sharex=True)

if len(channels) == 1:
    axes = [axes]

for ax, channel in zip(axes, channels):
    ax.plot(x, plot_df[channel], linewidth=0.7)
    ax.set_ylabel(channel)
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel(x_label)
fig.suptitle(f"5-Channel sEMG Signal: {plot_path.name}")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "five_channel_signal.png", dpi=150)
plt.close(fig)

print("\n저장 완료:", OUTPUT_DIR / "five_channel_signal.png")

# 같은 동작의 30도와 60도 비교
comparison_gesture = "fist"
comparison_files = {}

for _, row in summary.iterrows():
    if row["gesture"] == comparison_gesture and row["angle"] in (30, 60):
        comparison_files.setdefault(row["angle"], row["file"])

if 30 in comparison_files and 60 in comparison_files:
    df30, ch30 = read_emg(comparison_files[30])
    df60, ch60 = read_emg(comparison_files[60])

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=False)

    for ax, df, ch, angle in [
        (axes[0], df30, ch30, 30),
        (axes[1], df60, ch60, 60)
    ]:
        part = df.iloc[:5000]
        x = part["time"] if "time" in part.columns else range(len(part))
        ax.plot(x, part[ch[0]], linewidth=0.7)
        ax.set_title(f"{comparison_gesture}, elbow angle {angle} degrees")
        ax.set_ylabel(ch[0])
        ax.set_xlabel("Time / Sample")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "angle_comparison_30_60.png", dpi=150)
    plt.close(fig)

    print("저장 완료:", OUTPUT_DIR / "angle_comparison_30_60.png")
else:
    print("\n30도와 60도 비교 파일을 찾지 못했습니다.")

# 클래스별 파일 수 표 저장
class_counts = (
    summary.dropna(subset=["class"])
    .groupby("class")
    .size()
    .rename("file_count")
    .reset_index()
)

class_counts.to_csv(
    OUTPUT_DIR / "class_counts.csv",
    index=False,
    encoding="utf-8-sig"
)

print("저장 완료:", OUTPUT_DIR / "class_counts.csv")
print("\n실습 코드 실행이 끝났습니다.")
print("결과 폴더:", OUTPUT_DIR)
