"""Copy the trained onnx into Gothic so it preloads with the other voices."""
import shutil
from pathlib import Path

NAME = "pl_PL-bezimienny-medium"
SRC = Path(__file__).resolve().parents[1] / "weights" / "piper"
DST = Path(r"C:\Program Files (x86)\Steam\steamapps\common\Gothic II\system\piper_models")


def main():
    onnx = SRC / f"{NAME}.onnx"
    config = SRC / f"{NAME}.onnx.json"
    if not onnx.exists() or not config.exists():
        raise SystemExit(f"Missing {onnx}. Train first, then pull weights/piper.")
    DST.mkdir(parents=True, exist_ok=True)
    shutil.copy2(onnx, DST / onnx.name)
    shutil.copy2(config, DST / config.name)
    print(f"installed {DST / onnx.name}")
    print("Quit Gothic completely and start it again. The hero then loads with the other voices.")


if __name__ == "__main__":
    main()
