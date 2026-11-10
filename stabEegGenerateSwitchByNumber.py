# 拡張子を.pyに

# pip install pylsl numpy
import threading
import time
import numpy as np
from pylsl import StreamInfo, StreamOutlet

FS = 250
N_CH = 3

# 状態ごとの帯域プロファイル
PROFILES = {
    "1": {"label": "awake_focus", "delta": 0.05, "theta": 0.10, "alpha": 0.25, "beta": 0.45, "gamma": 0.15},
    "2": {"label": "relaxed",     "delta": 0.08, "theta": 0.15, "alpha": 0.45, "beta": 0.22, "gamma": 0.10},
    "3": {"label": "drowsy",      "delta": 0.15, "theta": 0.40, "alpha": 0.25, "beta": 0.12, "gamma": 0.08},
    "4": {"label": "n1",          "delta": 0.25, "theta": 0.45, "alpha": 0.15, "beta": 0.08, "gamma": 0.07},
    "5": {"label": "n2",          "delta": 0.35, "theta": 0.30, "alpha": 0.12, "beta": 0.13, "gamma": 0.10},
    "6": {"label": "n3",          "delta": 0.65, "theta": 0.20, "alpha": 0.06, "beta": 0.05, "gamma": 0.04},
}

state = PROFILES["1"]
state_lock = threading.Lock()

eeg_info = StreamInfo(
    "VIEStub", "EEG", N_CH + 1, FS, "float32", "vie_stub_eeg"
)
eeg_outlet = StreamOutlet(eeg_info)

band_info = StreamInfo(
    "VIEStub", "BAND", 5, 10, "float32", "vie_stub_band"
)
band_outlet = StreamOutlet(band_info)

rng = np.random.default_rng(42)


def input_loop():
    global state
    print("状態を選択してください:")
    print("1: 覚醒・集中")
    print("2: リラックス")
    print("3: 眠気")
    print("4: N1")
    print("5: N2（浅い睡眠）")
    print("6: N3（深い睡眠）")
    print("q: 終了")

    while True:
        key = input("> ").strip()

        if key == "q":
            print("終了します")
            break

        if key in PROFILES:
            with state_lock:
                state = PROFILES[key]
            print(f"状態を {state['label']} に切替")
        else:
            print("1〜6 または q を入力してください")


def eeg_loop():
    global state
    t = 0.0

    while True:
        with state_lock:
            current = state

        p = current

        # 複数帯域を重ねた擬似EEG
        eeg = (
            p["delta"] * 30 * np.sin(2 * np.pi * 2.0 * t)
            + p["theta"] * 20 * np.sin(2 * np.pi * 6.0 * t + 1.0)
            + p["alpha"] * 12 * np.sin(2 * np.pi * 10.0 * t + 2.0)
            + p["beta"] * 6 * np.sin(2 * np.pi * 20.0 * t + 3.0)
            + p["gamma"] * 3 * np.sin(2 * np.pi * 40.0 * t + 4.0)
        )

        # ゆっくりした振幅変調とノイズ
        eeg *= 1 + 0.2 * np.sin(2 * np.pi * 0.1 * t)
        eeg += rng.normal(0, 1.5)

        # N2用の睡眠紡錘波
        if p["label"] == "n2" and rng.random() < 0.002:
            duration = int(FS * 0.8)
            burst_t = np.arange(duration) / FS
            burst = np.sin(2 * np.pi * 13 * burst_t) * np.hanning(duration)
            eeg += 25 * burst

        # N3用の大きな徐波
        if p["label"] == "n3":
            eeg += 20 * np.sin(2 * np.pi * 1.2 * t)

        # 3チャンネル分は振幅を少し変えて送る
        samples = [eeg, eeg * 0.9, eeg * 1.1, 0]
        eeg_outlet.push_sample(samples)

        # BANDを10Hzで送信
        if int(t * FS) % 25 == 0:
            band = np.array([
                p["delta"], p["theta"], p["alpha"], p["beta"], p["gamma"]
            ])
            band += rng.normal(0, 0.02, size=5)
            band = np.clip(band, 0, 1)
            band /= band.sum()
            band_outlet.push_sample(band.tolist())

        t += 1 / FS
        time.sleep(1 / FS)


threading.Thread(target=input_loop, daemon=True).start()

try:
    eeg_loop()
except KeyboardInterrupt:
    pass