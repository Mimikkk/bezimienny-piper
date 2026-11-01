# Bezimienny Piper

Głos Bezimiennego jako model Piper (`pl_PL-bezimienny-medium`), ładowany tak samo jak Gosia. Tekst zostaje polski. Oryginalne kwestie z plików Speech gra dalej oryginałem (`ReplaceAllDialogues=0`). Nowe, nieudźwiękowione kwestie idą przez TTS.

## Co jest w tym repozytorium

- `weights/piper/pl_PL-bezimienny-medium.onnx` — nauczony głos, 800 epok
- `gtts/` — `gtts "tekst"` zapisuje `Documents/gtts/voice-line.wav`
- `gothic/piper_tts_server.py` — serwer z gry, z wyłączonym konwerterem RVC i z regułą wymowy
- `src/` — złożenie datasetu, trening, instalacja modelu do Gothica
- `vast/` — wgranie i start treningu na już wynajętej karcie. Skrypty same nie wypożyczają GPU
- `vendor/piper/` — piper_train z poprawkami pod Lightning 2 i eksport ONNX (CPU, `dynamo=False`)
- `datasets/piper/pl/bezimienny/metadata.csv` — pary id|tekst
- `training/piper/bezimienny/dataset.jsonl` i `config.json` — wynik preprocessu

## Czego git nie zawiera

GitHub nie przyjmuje plików ponad 100 MB, a nagrań z gry tu nie ma.

Zostają na dysku, w tym folderze albo w `Documents/code/rvc-tts-training`:

- `training/piper/bezimienny/checkpoints/last.ckpt` — epoka 799, punkt wznowienia (w tym folderze, ignorowany przez git)
- `datasets/piper/pl/bezimienny/wav/` — nagrania z gry
- `training/piper/bezimienny/cache/` — cache preprocessu

Wznowienie: wgraj projekt na `/workspace/rvc-tts-training` (takie ścieżki są w `dataset.jsonl`) i trenuj od `last.ckpt`.

## Wymowa

„i” zmiękcza „z” wewnątrz rdzenia (`zima`, `ziemia`). Po przedrostku `z-`, `bez-`, `roz-`, `przez-`, `wz-` „z” zostaje twarde, także gdy rdzeń zaczyna się od „im” (`Bezimienny`, `zimprowizować`). Rodzina `zim-` zostaje miękka, łącznie z `przezimować`.
