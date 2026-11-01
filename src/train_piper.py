"""Train a Piper medium voice and export pl_PL-bezimienny-medium.onnx.

The onnx is the same kind of file as pl_PL-gosia-medium. Drop both the onnx
and the json into Gothic's piper_models and the hero loads with the other voices.
"""
import os
import subprocess
import sys
from pathlib import Path

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "datasets" / "piper" / "pl" / "bezimienny"
TRAIN = ROOT / "training" / "piper" / "bezimienny"
OUT = ROOT / "weights" / "piper"
EPOCHS = int(os.environ.get("EPOCHS", "800"))
BATCH = int(os.environ.get("BATCH", "32"))
NAME = "pl_PL-bezimienny-medium"


def require_gpu():
    if not torch.cuda.is_available() or getattr(torch.version, "hip", None):
        raise SystemExit("GPU required. This run is for the rented NVIDIA card.")
    print(f"training on {torch.cuda.get_device_name(0)}", flush=True)


def preprocess():
    TRAIN.mkdir(parents=True, exist_ok=True)
    if (TRAIN / "dataset.jsonl").exists() and (TRAIN / "config.json").exists():
        print("preprocess already done", flush=True)
        return
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "piper_train.preprocess",
            "--language",
            "pl",
            "--input-dir",
            str(SRC),
            "--output-dir",
            str(TRAIN),
            "--dataset-format",
            "ljspeech",
            "--single-speaker",
            "--sample-rate",
            "22050",
            "--dataset-name",
            NAME,
            "--audio-quality",
            "medium",
            "--max-workers",
            os.environ.get("PREPROCESS_WORKERS", "8"),
        ]
    )


def main():
    require_gpu()
    if not (SRC / "metadata.csv").exists():
        raise SystemExit(f"Missing {SRC / 'metadata.csv'}. Run src/build_piper_dataset.py first.")
    preprocess()

    import json

    from piper_train.vits.lightning import VitsModel

    config = json.loads((TRAIN / "config.json").read_text(encoding="utf-8"))
    model = VitsModel(
        num_symbols=int(config["num_symbols"]),
        num_speakers=int(config["num_speakers"]),
        sample_rate=int(config["audio"]["sample_rate"]),
        dataset=[str(TRAIN / "dataset.jsonl")],
        batch_size=BATCH,
        validation_split=0.02,
        num_test_examples=4,
        num_workers=4,
    )
    ckpt = ModelCheckpoint(
        dirpath=str(TRAIN / "checkpoints"),
        filename="bezimienny-{epoch:04d}",
        monitor="val_loss",
        mode="min",
        every_n_epochs=50,
        save_top_k=2,
        save_last=True,
    )
    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        precision="16-mixed",
        max_epochs=EPOCHS,
        default_root_dir=str(TRAIN),
        callbacks=[ckpt],
        enable_progress_bar=True,
    )
    last = TRAIN / "checkpoints" / "last.ckpt"
    trainer.fit(model, ckpt_path=str(last) if last.exists() else None)

    checkpoints = sorted((TRAIN / "checkpoints").glob("*.ckpt"))
    if not checkpoints:
        raise SystemExit("Training finished without a checkpoint.")
    latest = max(checkpoints, key=lambda path: path.stat().st_mtime)
    OUT.mkdir(parents=True, exist_ok=True)
    onnx = OUT / f"{NAME}.onnx"
    subprocess.check_call(
        [sys.executable, "-m", "piper_train.export_onnx", str(latest), str(onnx)]
    )
    (OUT / f"{NAME}.onnx.json").write_text(
        (TRAIN / "config.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    print(f"exported {onnx}", flush=True)


if __name__ == "__main__":
    main()
