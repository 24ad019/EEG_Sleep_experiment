# sleep_detector.py

import time
import threading
from collections import deque
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf
from pylsl import StreamInlet, resolve_byprop
from scipy.signal import butter, filtfilt, welch

# =========================
# 設定
# =========================

FS = 600
WINDOW_SEC = 5
BAND_UPDATE_SEC = 1
CONFIRM_SEC = 15
GRACE_SEC = 60
ALARM_DELAY_SEC = 15 * 60

ALARM_PATH = "alarm.wav"

BASELINE_CAPTURE_SEC = 45

LOWCUT = 0.5
HIGHCUT = 40.0

BETA_QUALITY_FACTOR = 1.5
GAMMA_QUALITY_FACTOR = 1.5

SATURATION_LIMIT = 1_000_000
STEP_THRESHOLD = 200_000
AMPLITUDE_LIMIT = 500_000

# =========================
# LSL受信
# =========================

def resolve_stream():
    print("LSLストリームを探索中...")

    streams = resolve_byprop("type", "EEG", timeout=10)

    if not streams:
        raise RuntimeError("EEGタイプのLSLストリームが見つかりません")

    inlet = StreamInlet(streams[0], max_buflen=60)
    inlet.open_stream()

    info = inlet.info()
    fs = int(info.nominal_srate())
    ch_count = info.channel_count()

    print(f"接続: {info.name()}")
    print(f"サンプリングレート: {fs} Hz")
    print(f"チャンネル数: {ch_count}")

    return inlet, fs, ch_count


# =========================
# 前処理・特徴量
# =========================

def bandpass(x, fs):
    nyquist = fs / 2
    b, a = butter(
        4,
        [LOWCUT / nyquist, HIGHCUT / nyquist],
        btype="bandpass",
    )
    return filtfilt(b, a, x)


def band_power(x, fs):
    freqs, psd = welch(x, fs=fs, nperseg=fs * 2)

    def power(low, high):
        mask = (freqs >= low) & (freqs < high)
        return np.trapezoid(psd[mask], freqs[mask])

    return {
        "delta": power(0.5, 4),
        "theta": power(4, 8),
        "alpha": power(8, 13),
        "beta": power(13, 30),
        "gamma": power(30, 40),
    }


def relative_power(power):
    total = sum(power.values())

    if total <= 0:
        return {k: 0.0 for k in power}

    return {k: v / total for k, v in power.items()}


def is_low_quality(raw_window, baseline):
    x1 = raw_window[:, 0]
    x2 = raw_window[:, 1]

    for x in (x1, x2):
        if np.max(np.abs(x)) > SATURATION_LIMIT:
            return True

        if np.max(np.abs(np.diff(x))) > STEP_THRESHOLD:
            return True

        if np.max(np.abs(x - np.mean(x))) > AMPLITUDE_LIMIT:
            return True

    band = relative_power(
        band_power(bandpass(x1, FS), FS)
    )

    if band["beta"] > baseline["beta"] * BETA_QUALITY_FACTOR:
        return True

    if band["gamma"] > baseline["gamma"] * GAMMA_QUALITY_FACTOR:
        return True

    return False


# =========================
# 覚醒基準
# =========================

def capture_baseline(inlet, fs, ch_count):
    print(f"\n覚醒基準を取得します。")
    print(f"目を開け、安静にしてください。{BASELINE_CAPTURE_SEC}秒計測します。")

    for i in range(3, 0, -1):
        print(f"{i}...")
        time.sleep(1)

    samples = []
    expected = int(BASELINE_CAPTURE_SEC * fs)

    print("計測中...")

    while len(samples) < expected:
        chunk, _ = inlet.pull_chunk(timeout=1.0)

        if not chunk:
            continue

        for sample in chunk:
            validity = sample[0]
            contact = sample[2]

            if validity != 1:
                continue

            if contact == 0:
                continue

            samples.append(sample[3:3 + ch_count])

    raw = np.array(samples, dtype=float)

    print(f"有効サンプル数: {len(raw)}")

    baseline = {}

    for ch_index, name in enumerate(["uv1", "uv2"]):
        x = bandpass(raw[:, ch_index], fs)
        power = band_power(x, fs)
        rel = relative_power(power)

        baseline[name] = rel

        print(f"\n[{name}] 覚醒基準")
        for k, v in rel.items():
            print(f"  {k}: {v:.4f}")

    return baseline


# =========================
# 入眠判定
# =========================

def make_state_checker(baseline):
    def check(state):
        theta = state["theta"]
        alpha = state["alpha"]
        delta = state["delta"]
        beta = state["beta"]

        baseline_theta = baseline["theta"]
        baseline_alpha = baseline["alpha"]
        baseline_delta = baseline["delta"]
        baseline_beta = baseline["beta"]

        baseline_ratio = baseline_theta / max(baseline_alpha, 1e-9)
        current_ratio = theta / max(alpha, 1e-9)

        cond1 = (
            theta >= baseline_theta * 1.25
            and theta - baseline_theta >= 0.05
        )

        cond2 = (
            current_ratio >= baseline_ratio * 1.30
            and current_ratio - baseline_ratio >= 0.05
        )

        cond3 = (
            delta >= baseline_delta * 1.20
            and delta - baseline_delta >= 0.03
        )

        cond4 = alpha <= baseline_alpha * 0.85

        cond5 = beta <= baseline_beta * 0.90

        required = cond1 and cond2
        auxiliary = sum([cond3, cond4, cond5])

        return required and auxiliary >= 2

    return check


# =========================
# 音声再生
# =========================

def play_alarm(path):
    if not Path(path).exists():
        print(f"音声ファイルが見つかりません: {path}")
        return

    data, sr = sf.read(path, dtype="float32")

    if data.ndim > 1:
        data = data[:, 0]

    sd.play(data, sr)
    sd.wait()


# =========================
# メイン処理
# =========================

def main():
    inlet, fs, ch_count = resolve_stream()

    baseline = capture_baseline(inlet, fs, ch_count)

    is_sleep = make_state_checker(baseline)

    band_history = deque(maxlen=WINDOW_SEC)

    continuous_count = 0
    sleep_confirmed = False
    sleep_confirmed_at = None
    alarm_scheduled = False
    alarm_played = False

    print("\n監視を開始します。")
    print("Ctrl+Cで終了します。")

    try:
        while True:
            time.sleep(BAND_UPDATE_SEC)

            chunk, _ = inlet.pull_chunk(timeout=0.5)

            if not chunk:
                continue

            valid_samples = [
                s[3:3 + ch_count]
                for s in chunk
                if s[0] == 1 and s[2] != 0
            ]

            if not valid_samples:
                continue

            raw_window = np.array(valid_samples, dtype=float)

            if is_low_quality(raw_window, baseline["uv1"]):
                print("低品質窓: 判定をリセット")
                continuous_count = 0
                continue

            band = {
                "delta": 0.0,
                "theta": 0.0,
                "alpha": 0.0,
                "beta": 0.0,
                "gamma": 0.0,
            }

            for ch_index in range(ch_count):
                x = bandpass(raw_window[:, ch_index], fs)
                rel = relative_power(band_power(x, fs))

                for key in band:
                    band[key] += rel[key] / ch_count

            band_history.append(band)

            if len(band_history) < WINDOW_SEC:
                continue

            avg = {
                key: sum(b[key] for b in band_history) / len(band_history)
                for key in band
            }

            sleep_like = is_sleep(avg)

            if sleep_like:
                continuous_count += 1
            else:
                continuous_count = 0

            print(
                f"theta={avg['theta']:.3f} "
                f"alpha={avg['alpha']:.3f} "
                f"delta={avg['delta']:.3f} "
                f"beta={avg['beta']:.3f} "
                f"continuous={continuous_count}/{CONFIRM_SEC}"
            )

            if not sleep_confirmed and continuous_count >= CONFIRM_SEC:
                sleep_confirmed = True
                sleep_confirmed_at = time.time()
                alarm_scheduled = True

                print("\n入眠と判定しました。")
                print(f"{GRACE_SEC}秒の安定化猶予に入ります。")

            if sleep_confirmed:
                elapsed = time.time() - sleep_confirmed_at

                if elapsed < GRACE_SEC:
                    continue

                if not alarm_played:
                    print(f"\n{ALARM_DELAY_SEC}秒経過。音声を再生します。")
                    play_alarm(ALARM_PATH)
                    alarm_played = True

                if not sleep_like:
                    print("\n覚醒兆候を検出。入眠判定を取り消します。")
                    sleep_confirmed = False
                    sleep_confirmed_at = None
                    alarm_scheduled = False
                    alarm_played = False
                    continuous_count = 0

    except KeyboardInterrupt:
        print("\n終了します。")


if __name__ == "__main__":
    main()