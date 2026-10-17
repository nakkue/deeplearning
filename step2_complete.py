from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import iirnotch, butter, filtfilt, welch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parent
DATA = BASE / "sEMG-Hand-Gesture-Elbow-Angle-Dataset-main" / "data"
OUT = BASE / "results"
OUT.mkdir(exist_ok=True)

FS = 1000
NOTCH_HZ = 60
NOTCH_Q = 30
LOW_HZ, HIGH_HZ = 20, 450
SEGMENT_OFFSET = 1500
SEGMENT_LEN = 2000
WINDOW_LEN = 200
HOP_LEN = 100
ZC_THRESHOLD = 0.01
TEST_SIZE = 0.2
RANDOM_STATE = 42

plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def parse_file(path):
    name = path.stem.lower()
    m = re.search(r"_(30|60)\b", name)
    if not m:
        raise ValueError(f"팔꿈치 각도(30/60)를 파일명에서 찾지 못했습니다: {path.name}")
    angle = int(m.group(1))
    if "fist" in name:
        gesture = "fist"
    elif "open" in name:
        gesture = "open"
    elif "pro" in name:
        gesture = "pro"
    elif "extor" in name:
        gesture = "extor"
    else:
        raise ValueError(f"동작 이름을 파일명에서 찾지 못했습니다: {path.name}")
    return {"path": path, "filename": path.name, "angle": angle,
            "gesture": gesture, "class": f"{angle}_{gesture}"}


def read_file(path):
    df = pd.read_csv(path)
    channels = [c for c in df.columns if str(c).lower().startswith("bandpass")]
    if len(channels) != 5:
        raise ValueError(f"{path.name}: bandpass 채널 5개를 찾지 못했습니다: {channels}")
    for c in channels + [c for c in ("time", "onset", "mark") if c in df.columns]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=channels).reset_index(drop=True)
    x = df[channels].to_numpy(dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError(f"{path.name}: 신호에 NaN/무한대가 있습니다.")
    if "time" in df.columns and len(df) > 2:
        dt = np.median(np.diff(df["time"].to_numpy(dtype=float)))
        if dt > 0:
            observed_fs = 1 / dt
            if abs(observed_fs - FS) > 5:
                raise ValueError(f"{path.name}: 샘플링 주파수가 {observed_fs:.2f}Hz입니다. FS 설정을 확인하세요.")
    return df, x, channels


def filter_signal(x):
    """60Hz notch 필터 후 20~450Hz Butterworth band-pass 필터."""
    bn, an = iirnotch(NOTCH_HZ, NOTCH_Q, fs=FS)
    notched = filtfilt(bn, an, x, axis=0)
    bb, ab = butter(4, [LOW_HZ, HIGH_HZ], btype="bandpass", fs=FS)
    filtered = filtfilt(bb, ab, notched, axis=0)
    return filtered


def get_mark_positions(df, filename):
    if "mark" not in df.columns:
        raise ValueError(f"{filename}: mark 열이 없습니다.")
    marks = np.flatnonzero(df["mark"].fillna(0).to_numpy() != 0)
    if len(marks) == 0:
        raise ValueError(f"{filename}: mark 값이 모두 0입니다.")
    return marks


def cut_segment(filtered, mark_pos, filename):
    # 강의 실습 가정: 각 mark 기준점에서 1500 samples 뒤부터 2초 구간
    start = int(mark_pos) + SEGMENT_OFFSET
    end = start + SEGMENT_LEN
    if end > len(filtered):
        return None
    segment = filtered[start:end]
    if segment.shape != (SEGMENT_LEN, 5):
        raise ValueError(f"{filename}: 잘라낸 구간 모양이 잘못됐습니다: {segment.shape}")
    return segment


def make_windows(segment):
    count = (len(segment) - WINDOW_LEN) // HOP_LEN + 1
    return np.stack([
        segment[i * HOP_LEN:i * HOP_LEN + WINDOW_LEN]
        for i in range(count)
    ])


def extract_features(window):
    """(200, 5) 신호를 RMS/MAV/WL/ZC/SSC 각 채널별로 계산해 25차원으로 만든다."""
    w = np.asarray(window, dtype=np.float64)
    if w.ndim != 2 or w.shape[1] != 5:
        raise ValueError(f"윈도우는 (샘플 수, 5채널)이어야 합니다: {w.shape}")
    rms = np.sqrt(np.mean(w ** 2, axis=0))
    mav = np.mean(np.abs(w), axis=0)
    diff = np.diff(w, axis=0)
    wl = np.sum(np.abs(diff), axis=0)
    zc = np.sum(
        ((w[:-1] * w[1:]) < 0) & (np.abs(diff) >= ZC_THRESHOLD),
        axis=0
    )
    ssc = np.sum((diff[:-1] * diff[1:]) < 0, axis=0)
    result = np.concatenate([rms, mav, wl, zc.astype(float), ssc.astype(float)])
    if result.shape != (25,) or not np.isfinite(result).all():
        raise ValueError("특징 벡터가 25차원이 아니거나 유효하지 않습니다.")
    return result


def save_filter_plot(raw, filtered, filename):
    n = min(len(raw), FS * 20)
    fr, pr = welch(raw[:n, 0], fs=FS, nperseg=min(4096, n))
    ff, pf = welch(filtered[:n, 0], fs=FS, nperseg=min(4096, n))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.semilogy(fr, pr + 1e-15, label="Before filtering")
    ax.semilogy(ff, pf + 1e-15, label="After filtering")
    ax.axvline(60, linestyle="--", label="60 Hz")
    ax.axvline(20, linestyle=":", label="20-450 Hz band")
    ax.axvline(450, linestyle=":")
    ax.set_xlim(0, 500)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power spectral density")
    ax.set_title(f"Filter check: {filename}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "filter_check.png", dpi=160)
    plt.close(fig)


def save_signal_plots(raw, filtered, segment, windows, filename):
    n = min(3000, len(raw))
    t = np.arange(n) / FS
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(t, raw[:n, 0], linewidth=0.6)
    axes[0].set_title(f"Raw signal: {filename}, channel 1")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(t, filtered[:n, 0], linewidth=0.6)
    axes[1].set_title("Filtered signal, channel 1")
    axes[1].set_xlabel("Time from file start (s)")
    axes[1].set_ylabel("Amplitude")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "signal_example.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(np.arange(len(segment)) / FS, segment[:, 0], label="2-second segment", linewidth=0.7)
    ax.plot(np.arange(WINDOW_LEN) / FS, windows[0, :, 0], label="First 200 ms window", linewidth=1.2)
    ax.set_title("Segment and first window, channel 1")
    ax.set_xlabel("Time within segment (s)")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "window_check.png", dpi=160)
    plt.close(fig)


def create_features(records):
    features, labels, detail = [], [], []
    for rec in records:
        df, raw, _ = read_file(rec["path"])
        filtered = filter_signal(raw)
        marks = get_mark_positions(df, rec["filename"])
        valid_segments = 0
        created_windows = 0
        skipped = 0
        for trial_num, mark_pos in enumerate(marks, start=1):
            segment = cut_segment(filtered, int(mark_pos), rec["filename"])
            if segment is None:
                skipped += 1
                continue
            windows = make_windows(segment)
            if windows.shape != (19, 200, 5):
                raise ValueError(f"{rec['filename']}: 윈도우 모양 오류 {windows.shape}")
            for window in windows:
                features.append(extract_features(window))
                labels.append(rec["class"])
            valid_segments += 1
            created_windows += len(windows)
        detail.append({
            "filename": rec["filename"], "class": rec["class"],
            "mark_count": len(marks), "valid_segments": valid_segments,
            "windows_created": created_windows, "skipped_incomplete": skipped
        })
    if not features:
        raise RuntimeError("특징이 생성되지 않았습니다. mark 위치와 구간 길이를 확인하세요.")
    return np.vstack(features), np.asarray(labels), pd.DataFrame(detail)


def main():
    print("=" * 60)
    print("2주차 sEMG 전처리 및 특징 추출 실습")
    print("=" * 60)
    if not DATA.exists():
        raise FileNotFoundError(f"데이터 폴더를 찾지 못했습니다: {DATA}")
    paths = sorted(DATA.glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"CSV 파일이 없습니다: {DATA}")
    records = [parse_file(p) for p in paths]
    summary = pd.DataFrame([{
        "filename": r["filename"], "class": r["class"],
        "angle": r["angle"], "gesture": r["gesture"]
    } for r in records])
    print(f"CSV 파일 수: {len(records)}")
    print("\n클래스별 파일 수:")
    print(summary.groupby("class").size().to_string())

    # 첫 파일로 필터/윈도우 결과 그림 저장
    sample = records[0]
    df, raw, channels = read_file(sample["path"])
    filtered = filter_signal(raw)
    marks = get_mark_positions(df, sample["filename"])
    first_segment = cut_segment(filtered, int(marks[0]), sample["filename"])
    if first_segment is None:
        raise RuntimeError("첫 mark 뒤에 1.5초 + 2초 구간이 부족합니다.")
    first_windows = make_windows(first_segment)
    print("\n필터: 60Hz notch + 20~450Hz band-pass")
    print("샘플링 주파수:", FS, "Hz")
    print("채널:", channels)
    print("첫 파일 mark 개수:", len(marks), "| 첫 mark 위치:", int(marks[0]))
    print("세그먼트 모양:", first_segment.shape)
    print("윈도우 설정: 200 samples (200 ms), hop 100 samples (100 ms)")
    print("윈도우 모양:", first_windows.shape, "(윈도우 수, 샘플 수, 채널 수)")
    print("첫 윈도우 특징 차원:", extract_features(first_windows[0]).shape)
    save_filter_plot(raw, filtered, sample["filename"])
    save_signal_plots(raw, filtered, first_segment, first_windows, sample["filename"])

    # 중요: 윈도우를 만들기 전에 CSV 파일 단위로 train/test 분리
    labels_by_file = np.asarray([r["class"] for r in records])
    train_records, test_records = train_test_split(
        records, test_size=TEST_SIZE, random_state=RANDOM_STATE,
        stratify=labels_by_file
    )
    train_files = {r["filename"] for r in train_records}
    test_files = {r["filename"] for r in test_records}
    if train_files & test_files:
        raise RuntimeError("파일 단위 분할에 누수가 있습니다.")
    print("\n파일 단위 분할:")
    print("학습 파일:", len(train_records), "| 테스트 파일:", len(test_records))
    print("학습/테스트 파일 중복:", len(train_files & test_files))

    X_train, y_train, train_detail = create_features(train_records)
    X_test, y_test, test_detail = create_features(test_records)
    class_names = sorted(summary["class"].unique())
    class_to_id = {name: i for i, name in enumerate(class_names)}
    y_train_id = np.asarray([class_to_id[y] for y in y_train], dtype=np.int64)
    y_test_id = np.asarray([class_to_id[y] for y in y_test], dtype=np.int64)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\n데이터셋 모양:")
    print("X_train:", X_train_scaled.shape, "y_train:", y_train_id.shape)
    print("X_test :", X_test_scaled.shape, "y_test :", y_test_id.shape)
    print("표준화: 학습 데이터에만 fit, 테스트 데이터는 transform만 수행")

    np.savez_compressed(
        OUT / "dataset.npz",
        X_train=X_train_scaled.astype(np.float32),
        y_train=y_train_id,
        X_test=X_test_scaled.astype(np.float32),
        y_test=y_test_id,
        class_names=np.asarray(class_names),
        feature_names=np.asarray(
            [f"{f}_ch{c}" for f in ["RMS", "MAV", "WL", "ZC", "SSC"] for c in range(1, 6)]
        ),
        train_files=np.asarray(sorted(train_files)),
        test_files=np.asarray(sorted(test_files)),
        fs=np.asarray(FS),
        window_len=np.asarray(WINDOW_LEN),
        hop_len=np.asarray(HOP_LEN)
    )

    train_counts = pd.Series(y_train).value_counts().reindex(class_names, fill_value=0)
    test_counts = pd.Series(y_test).value_counts().reindex(class_names, fill_value=0)
    counts = pd.DataFrame({
        "class": class_names,
        "train_windows": [int(train_counts[c]) for c in class_names],
        "test_windows": [int(test_counts[c]) for c in class_names],
        "total_windows": [int(train_counts[c] + test_counts[c]) for c in class_names]
    })
    counts.to_csv(OUT / "dataset_summary.csv", index=False, encoding="utf-8-sig")
    pd.concat([
        train_detail.assign(split="train"),
        test_detail.assign(split="test")
    ], ignore_index=True).to_csv(
        OUT / "trial_file_summary.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(X_train[:10], columns=[
        f"{f}_ch{c}" for f in ["RMS", "MAV", "WL", "ZC", "SSC"] for c in range(1, 6)
    ]).assign(class_name=y_train[:10]).to_csv(
        OUT / "feature_preview.csv", index=False, encoding="utf-8-sig"
    )

    print("\n클래스별 윈도우 수:")
    print(counts.to_string(index=False))
    print("\n생성된 파일:")
    for name in [
        "filter_check.png", "signal_example.png", "window_check.png",
        "dataset.npz", "dataset_summary.csv", "trial_file_summary.csv",
        "feature_preview.csv"
    ]:
        p = OUT / name
        print(f"{name}: {'OK' if p.exists() else 'MISSING'}" +
              (f" ({p.stat().st_size:,} bytes)" if p.exists() else ""))
    print("\n2주차 전처리 및 특징 추출 단계 완료.")


if __name__ == "__main__":
    main()
