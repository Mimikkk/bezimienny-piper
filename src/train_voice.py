"""Train the Bezimienny RVC v2 timbre on whatever CUDA GPU this process sees.

The wavs in datasets/voices/pl/male/bezimienny are the original performance.
HuBERT features match the local Gothic converter (transformers, layer 12).
"""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import HubertModel

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor" / "rvc"
WAVS = ROOT / "datasets" / "voices" / "pl" / "male" / "bezimienny"
EXP = VENDOR / "logs" / "bezimienny"
PRE_G = VENDOR / "assets" / "pretrained_v2" / "f0G40k.pth"
PRE_D = VENDOR / "assets" / "pretrained_v2" / "f0D40k.pth"
EPOCHS = os.environ.get("EPOCHS", "100")
BATCH = os.environ.get("BATCH", "24")
SAVE_EVERY = os.environ.get("SAVE_EVERY", "10")


def require_gpu():
    if not torch.cuda.is_available() or getattr(torch.version, "hip", None):
        raise SystemExit("GPU required. This run is for the rented NVIDIA card.")
    name = torch.cuda.get_device_name(0)
    print(f"training on {name}", flush=True)
    return name


def run(script, args):
    cmd = [sys.executable, script, *args]
    print(" ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=VENDOR)


def extract_hubert():
    wav_dir = EXP / "1_16k_wavs"
    out_dir = EXP / "3_feature768"
    out_dir.mkdir(parents=True, exist_ok=True)
    model = HubertModel.from_pretrained("facebook/hubert-base-ls960").cuda().eval()
    names = sorted(p.name for p in wav_dir.glob("*.wav"))
    print(f"hubert {len(names)}", flush=True)
    for i, name in enumerate(names):
        dest = out_dir / (name[:-4] + ".npy")
        if dest.exists():
            continue
        wav, sr = sf.read(wav_dir / name, dtype="float32")
        if wav.ndim == 2:
            wav = wav.mean(axis=1)
        if sr != 16000:
            raise RuntimeError(f"{name} sr={sr}")
        feats = torch.from_numpy(np.ascontiguousarray(wav)).float().view(1, -1).cuda()
        with torch.no_grad():
            hidden = model(feats, output_hidden_states=True).hidden_states[12]
        np.save(dest, hidden.squeeze(0).float().cpu().numpy())
        if i % 50 == 0:
            print(f"hubert {i}/{len(names)}", flush=True)
    write_mute(model)
    del model
    torch.cuda.empty_cache()


def write_mute(model):
    mute = VENDOR / "logs" / "mute"
    wav_dir = mute / "0_gt_wavs"
    fea_dir = mute / "3_feature768"
    f0_dir = mute / "2a_f0"
    nsf_dir = mute / "2b-f0nsf"
    for folder in (wav_dir, fea_dir, f0_dir, nsf_dir):
        folder.mkdir(parents=True, exist_ok=True)
    sf.write(wav_dir / "mute40k.wav", np.zeros(40000, dtype=np.float32), 40000, subtype="PCM_16")
    wav16 = np.zeros(16000, dtype=np.float32)
    feats = torch.from_numpy(wav16).view(1, -1).cuda()
    with torch.no_grad():
        hidden = model(feats, output_hidden_states=True).hidden_states[12]
    np.save(fea_dir / "mute.npy", hidden.squeeze(0).float().cpu().numpy())
    n = wav16.shape[0] // 160
    f0 = np.zeros(n, dtype=np.float32)
    coarse = np.ones(n, dtype=np.int32)
    np.save(nsf_dir / "mute.wav.npy", f0)
    np.save(f0_dir / "mute.wav.npy", coarse)


def write_config_and_filelist():
    import json
    from random import shuffle

    cfg_path = VENDOR / "configs" / "v1" / "40k.json"
    with cfg_path.open(encoding="utf-8") as handle:
        cfg = json.load(handle)
    cfg["train"]["fp16_run"] = True
    cfg["train"]["log_interval"] = 20
    with (EXP / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(cfg, handle, indent=4, sort_keys=True)
        handle.write("\n")

    gt = EXP / "0_gt_wavs"
    fea = EXP / "3_feature768"
    f0 = EXP / "2a_f0"
    nsf = EXP / "2b-f0nsf"
    names = (
        {p.name.split(".")[0] for p in gt.glob("*.wav")}
        & {p.name.split(".")[0] for p in fea.glob("*.npy")}
        & {p.name.split(".")[0] for p in f0.glob("*.npy")}
        & {p.name.split(".")[0] for p in nsf.glob("*.npy")}
    )
    rows = []
    for name in names:
        rows.append(
            "|".join(
                [
                    (gt / f"{name}.wav").as_posix(),
                    (fea / f"{name}.npy").as_posix(),
                    (f0 / f"{name}.wav.npy").as_posix(),
                    (nsf / f"{name}.wav.npy").as_posix(),
                    "0",
                ]
            )
        )
    mute_root = (VENDOR / "logs" / "mute").as_posix()
    for _ in range(2):
        rows.append(
            "|".join(
                [
                    f"{mute_root}/0_gt_wavs/mute40k.wav",
                    f"{mute_root}/3_feature768/mute.npy",
                    f"{mute_root}/2a_f0/mute.wav.npy",
                    f"{mute_root}/2b-f0nsf/mute.wav.npy",
                    "0",
                ]
            )
        )
    shuffle(rows)
    (EXP / "filelist.txt").write_text("\n".join(rows), encoding="utf-8")
    print(f"filelist {len(rows)}", flush=True)


def collect_weights():
    out = ROOT / "weights" / "bezimienny"
    out.mkdir(parents=True, exist_ok=True)
    src = VENDOR / "assets" / "weights"
    copied = 0
    if src.exists():
        for path in src.glob("bezimienny*.pth"):
            target = out / path.name
            target.write_bytes(path.read_bytes())
            copied += 1
            print(f"saved {target}", flush=True)
    if copied == 0:
        raise SystemExit(f"no weights written in {src}")


def main():
    require_gpu()
    if not any(WAVS.glob("*.wav")):
        raise SystemExit(f"no wavs in {WAVS}")
    for path in (PRE_G, PRE_D):
        if not path.exists() or path.stat().st_size < 1000000:
            raise SystemExit(f"missing pretrained weight {path}")
    EXP.mkdir(parents=True, exist_ok=True)
    cpus = str(max(1, os.cpu_count() or 1))
    run(
        "infer/modules/train/preprocess.py",
        [WAVS.as_posix(), "40000", cpus, EXP.as_posix(), "False", "3.0"],
    )
    run(
        "infer/modules/train/extract/extract_f0_print.py",
        [EXP.as_posix(), cpus, "pm"],
    )
    extract_hubert()
    write_config_and_filelist()
    (VENDOR / "assets" / "weights").mkdir(parents=True, exist_ok=True)
    run(
        "infer/modules/train/train.py",
        [
            "-e",
            "bezimienny",
            "-sr",
            "40k",
            "-f0",
            "1",
            "-bs",
            BATCH,
            "-g",
            "0",
            "-te",
            EPOCHS,
            "-se",
            SAVE_EVERY,
            "-pg",
            PRE_G.as_posix(),
            "-pd",
            PRE_D.as_posix(),
            "-l",
            "1",
            "-c",
            "0",
            "-sw",
            "1",
            "-v",
            "v2",
        ],
    )
    collect_weights()


if __name__ == "__main__":
    main()
