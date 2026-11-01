"""gtts "tekst" -> Documents/gtts/voice-line.wav

Synthesizes with the Bezimienny Piper onnx. No RVC converter.
"""
import re
import sys
import wave
from pathlib import Path

from piper import PiperVoice
from piper.config import SynthesisConfig

_VOWELS = set("aąeęioóuy")
_PREFIXES = ("przez", "bez", "roz", "wz", "z")
_WORD = re.compile(r"[A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż]+")
# zima / zimno / zimowy, including przezimować. That z is soft.
_ZIM = re.compile(
    r"(?i)^zim(?:"
    r"a|ą|ę|y|ie|ach|ami|om|"
    r"n(?:o|a|ą|e|y|ym|ych|ymi|ego|emu|i|ie|ej|ość|ości|ością|iej|iejsz[a-ząćęłńóśźż]*)?|"
    r"ow(?:y|a|ą|e|i|o|ym|ych|ymi|ego|ej|emu|"
    r"ać|ał(?:a|o|y|am|aś|eś|em)?|"
    r"anie|ania|aniu|aniem|aniom|aniami|aniach|"
    r"isk[a-ząćęłńóśźż]*|ość[a-ząćęłńóśźż]*)?"
    r")$"
)
# One-letter z- is also how loans start (Zimbabwe, zip). Split those only
# when the stem is an ordinary Polish word.
_POLISH_STEM = re.compile(
    r"(?i).*(?:ować|ywać|iwać|nąć|ać|eć|ić|yć|"
    r"owan[a-ząćęłńóśźż]*|ując[a-ząćęłńóśźż]*|"
    r"ny|na|ne|ni|ną|nym|nych|nymi|nego|nemu|"
    r"owy|owa|owe|owi|owym|ową|owych|"
    r"ski|ska|skie|ość|ości|"
    r"ł|ła|ło|li|łem|łam|"
    r"uj|yj|my|cie|sz)$"
)


def fix_polish_z(text):
    """Hard z after z-, bez-, roz-, przez-, wz- when the next morpheme starts with i.

    espeak already does that, except the stem starts with im: the letters spell zim,
    so it uses the soft z of zima. The winter root stays soft.
    """
    return _WORD.sub(_fix_word, text)


def _fix_word(match):
    word = match.group(0)
    lower = word.lower()
    head = 3 if lower.startswith("nie") and any(lower.startswith(p, 3) for p in _PREFIXES) else 0
    for prefix in _PREFIXES:
        if not lower.startswith(prefix, head):
            continue
        rest = lower[head + len(prefix):]
        if len(rest) < 2 or rest[0] != "i" or rest[1] in _VOWELS:
            continue
        if _ZIM.match(lower[head + len(prefix) - 1:]):
            return word
        if prefix == "z" and not _POLISH_STEM.match(rest):
            continue
        cut = head + len(prefix)
        return word[:cut] + "'" + word[cut:]
    return word


OUT = Path(r"C:\Users\mimi\Documents\gtts\voice-line.wav")
MODEL = Path(
    r"C:\Program Files (x86)\Steam\steamapps\common\Gothic II"
    r"\system\piper_models\pl_PL-bezimienny-medium.onnx"
)


def main():
    text = " ".join(sys.argv[1:]).strip()
    if not text:
        print('Uzycie: gtts "tekst"', file=sys.stderr)
        return 1
    if not MODEL.exists() or not Path(str(MODEL) + ".json").exists():
        print(f"Brak modelu: {MODEL}", file=sys.stderr)
        return 1

    voice = PiperVoice.load(str(MODEL), config_path=str(MODEL) + ".json")
    config = SynthesisConfig(length_scale=1.0, noise_scale=0.667, noise_w_scale=0.8)
    pcm = b"".join(
        chunk.audio_int16_bytes
        for chunk in voice.synthesize(fix_polish_z(text), syn_config=config)
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(OUT), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(voice.config.sample_rate)
        handle.writeframes(pcm)
    print(OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
