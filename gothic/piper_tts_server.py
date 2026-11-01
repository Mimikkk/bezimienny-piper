#!/usr/bin/env python3
"""
Piper-TTS HTTP Server for Gothic Union Plugin (ASYNC VERSION)
Uses threading for async TTS generation with abort support

Architecture:
- POST /generate → Returns request_id, starts generation in thread
- GET /status/<id> → Returns status (generating/ready/aborted/error)
- GET /audio/<id> → Returns audio data (if ready)
- POST /abort/<id> → Aborts generation
- GET /health → Health check
- GET /download_status → Multi-model download status
"""

import argparse
import io
import logging
import os
import re
import subprocess
import sys
import time
import threading
import urllib.request
import uuid
import wave
from pathlib import Path
from datetime import datetime, timedelta

import yaml
from flask import Flask, request, send_file, jsonify
from piper import PiperVoice

# Setup logging
log_file = Path(__file__).parent / 'piper_tts_server.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='w', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

# Fix console encoding for Windows
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        pass

logger = logging.getLogger(__name__)

logger.info("=" * 60)
logger.info("Piper-TTS Server (ASYNC)")
logger.info("=" * 60)

# Verify environment
import os
logger.info("Environment Verification:")
logger.info(f"Python Executable: {sys.executable}")
logger.info(f"Python Version: {sys.version}")
logger.info(f"Python Prefix: {sys.prefix}")
logger.info("=" * 60)

# Flask app
app = Flask(__name__)

# Voice cache (now supports multiple models per language)
voice_cache = {}
voice_lock = threading.Lock()  # Thread-safe voice access

# Configuration loaded from YAML
config = None
config_lock = threading.Lock()

def load_config():
    """Load voice mapping configuration from YAML"""
    global config
    
    # piper_tts_server.py is in System/, so config is in same directory
    config_path = Path(__file__).parent / 'Union_RealTime_TTS_Voice_Mapping.yaml'
    
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        logger.error("Please create System/Union_RealTime_TTS_Voice_Mapping.yaml")
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    logger.info(f"Loading configuration from: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Configuration loaded (version: {config.get('version', 'unknown')})")
    
    # Log available languages
    languages = config.get('languages', {})
    logger.info(f"Available languages: {', '.join(languages.keys())}")
    
    # Log German voice mappings
    if 'de' in languages:
        de_config = languages['de']
        voice_mapping = de_config.get('voice_mapping', {})
        logger.info(f"German voice mappings: {len(voice_mapping)} entries")
        for voice_id, model_name in voice_mapping.items():
            logger.info(f"  Voice ID {voice_id} → {model_name}")
    
    return config

def get_model_for_voice(language: str, voice_id: int = None, gender: int = 0) -> str:
    """Get model name for a specific language, voice ID, and gender
    
    Args:
        language: Language code (e.g., 'de', 'en')
        voice_id: NPC voice ID (optional)
        gender: NPC gender (0=MALE, 1=FEMALE)
    
    Returns:
        Model name to use for TTS
    
    Female voices:
        - If the language defines female_models, pick one from that list by voice id
        - Otherwise keep the German fallback (voice id 16/17 stay mapped, others alternate ID16/ID17)
    """
    with config_lock:
        if not config:
            raise RuntimeError("Configuration not loaded")
        
        lang_config = config.get('languages', {}).get(language)
        if not lang_config:
            logger.warning(f"Language '{language}' not found in config, using 'de' as fallback")
            lang_config = config.get('languages', {}).get('de', {})
        
        female_models = lang_config.get('female_models') or []
        male_models = lang_config.get('male_models') or []
        hero_model = lang_config.get('hero_model')
        hero_voice_ids = {int(v) for v in (lang_config.get('hero_voice_ids') or [])}

        # Language-specific female pool. Same voice id always gets the same model.
        if gender == 1 and female_models:
            index = (voice_id or 0) % len(female_models)
            model_name = female_models[index]
            logger.info(f"Female NPC (voice_id={voice_id}) -> {model_name}")
            return model_name

        # Legacy German fallback when this language has no female pool.
        if gender == 1 and voice_id not in [16, 17]:
            last_digit = (voice_id or 0) % 10
            if last_digit <= 5:
                female_fallback = "de_DE-ID16"
            else:
                female_fallback = "de_DE-ID17"
            logger.info(f"Female NPC detected (voice_id={voice_id}, last_digit={last_digit}, gender={gender}), using female fallback: {female_fallback}")
            return female_fallback

        # Bezimienny keeps one dedicated voice, separate from the male NPC pool.
        # His own Piper file, once trained, loads like Gosia. Until that file
        # is in piper_models, keep the current carrier.
        if voice_id in hero_voice_ids:
            native = native_hero_onnx()
            if native:
                logger.info(f"Hero voice ID {voice_id} -> {native}")
                return native
            if hero_model:
                logger.info(f"Hero voice ID {voice_id} -> {hero_model}")
                return hero_model

        # Check voice mapping (only for languages that support it)
        if voice_id is not None:
            voice_mapping = lang_config.get('voice_mapping', {})
            if voice_id in voice_mapping:
                model_name = voice_mapping[voice_id]
                logger.info(f"Voice ID {voice_id} mapped to model: {model_name}")
                return model_name

        if male_models:
            index = (voice_id or 0) % len(male_models)
            model_name = male_models[index]
            logger.info(f"Male NPC (voice_id={voice_id}) -> {model_name}")
            return model_name

        # Fallback to default model
        default_model = lang_config.get('default_model')
        if not default_model:
            raise ValueError(f"No default model defined for language: {language}")
        
        if voice_id is not None:
            logger.info(f"Voice ID {voice_id} not mapped, using default: {default_model}")
        
        return default_model

def get_all_models_for_language(language: str = None) -> dict:
    """Get all models defined for a language (or all models if language=None)"""
    with config_lock:
        if not config:
            raise RuntimeError("Configuration not loaded")
        
        if language is None:
            # Return all models from all languages
            all_models = {}
            for lang, lang_config in config.get('languages', {}).items():
                models = lang_config.get('models', {})
                all_models.update(models)
            return all_models
        
        lang_config = config.get('languages', {}).get(language, {})
        return lang_config.get('models', {})

# Download status tracking
class DownloadStatus:
    IDLE = "idle"
    DOWNLOADING = "downloading"
    READY = "ready"
    ERROR = "error"

# Global download tracking
download_status = {}
download_lock = threading.Lock()
download_queue = []  # List of (model_name, model_info) tuples
current_download_index = 0
total_downloads = 0

# Request storage
class RequestStatus:
    LOADING_MODEL = "loading_model"  # NEW: Model is being loaded from disk
    GENERATING = "generating"
    READY = "ready"
    ABORTED = "aborted"
    ERROR = "error"

class TTSRequest:
    def __init__(self, request_id, text, language, voice_id=None, gender=0, length_scale=1.0, volume=0.5):
        self.request_id = request_id
        self.text = text
        self.language = language
        self.voice_id = voice_id
        self.gender = gender  # 0=MALE, 1=FEMALE
        self.length_scale = length_scale  # Speed control (1.0=normal, <1.0=faster, >1.0=slower)
        self.volume = volume  # Volume control (0.0-1.0, default 0.5 = -24dB, 21 steps)
        self.status = RequestStatus.GENERATING
        self.progress = 0
        self.audio_data = None
        self.error_message = None
        self.created_at = datetime.now()
        self.abort_flag = threading.Event()
        self.model_name = None  # NEW: Track which model is being used
    
    def should_abort(self):
        return self.abort_flag.is_set()
    
    def abort(self):
        self.abort_flag.set()
        self.status = RequestStatus.ABORTED

# Global request storage
requests_storage = {}
storage_lock = threading.Lock()

def download_voice_model(model_name: str, model_info: dict, index: int, total: int):
    """Download a single voice model"""
    import urllib.request
    
    global current_download_index
    
    with download_lock:
        if model_name not in download_status:
            download_status[model_name] = {
                'status': DownloadStatus.IDLE,
                'progress': 0,
                'total_mb': model_info.get('size_mb', 50),
                'downloaded_mb': 0,
                'error': None,
                'index': index,
                'total': total
            }
        current_download_index = index
    
    try:
        logger.info(f"[{index}/{total}] Starting download: {model_name}")
        logger.info(f"Description: {model_info.get('description', 'N/A')}")
        
        url_onnx = model_info.get('url_onnx')
        url_json = model_info.get('url_json')
        
        if not url_onnx or not url_json:
            raise ValueError(f"Missing download URLs for model: {model_name}")
        
        # Extract actual filenames from URLs (they might differ from model_name)
        onnx_filename = url_onnx.split('/')[-1]
        json_filename = url_json.split('/')[-1]
        
        # Determine file paths using actual filenames from URLs
        model_file = Path('piper_models') / onnx_filename
        config_file = Path('piper_models') / json_filename
        
        logger.info(f"Downloading to: {model_file}")
        logger.info(f"Config file: {config_file}")
        
        # Create models directory
        model_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Get actual file size from HTTP headers
        logger.info(f"Downloading {url_onnx}")
        req = urllib.request.Request(url_onnx, method='HEAD')
        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                content_length = response.headers.get('Content-Length')
                if content_length:
                    total_bytes = int(content_length)
                    total_mb = total_bytes / 1024 / 1024
                    logger.info(f"Actual file size: {total_mb:.2f} MB")
                    
                    # Update status with actual size
                    with download_lock:
                        download_status[model_name]['total_mb'] = total_mb
                else:
                    total_mb = model_info.get('size_mb', 50)
                    logger.warning(f"Could not determine file size, using estimate: {total_mb} MB")
        except Exception as e:
            logger.warning(f"Could not get file size: {e}")
            total_mb = model_info.get('size_mb', 50)
        
        # Update status
        with download_lock:
            download_status[model_name]['status'] = DownloadStatus.DOWNLOADING
            download_status[model_name]['progress'] = 0
            download_status[model_name]['downloaded_mb'] = 0
        
        # Download .onnx file
        def report_progress(block_num, block_size, total_size):
            if total_size > 0:
                progress = min(int((block_num * block_size / total_size) * 50), 50)  # 0-50%
                downloaded_bytes = block_num * block_size
                downloaded_mb = downloaded_bytes / 1024 / 1024
                with download_lock:
                    download_status[model_name]['progress'] = progress
                    download_status[model_name]['downloaded_mb'] = downloaded_mb
        
        urllib.request.urlretrieve(url_onnx, str(model_file), reporthook=report_progress)
        logger.info(f"Downloaded .onnx file ({model_file.stat().st_size / 1024 / 1024:.2f} MB)")
        
        # Download .onnx.json file
        logger.info(f"Downloading {url_json}")
        
        def report_progress_json(block_num, block_size, total_size):
            if total_size > 0:
                progress = 50 + min(int((block_num * block_size / total_size) * 50), 50)  # 50-100%
                with download_lock:
                    download_status[model_name]['progress'] = progress
        
        urllib.request.urlretrieve(url_json, str(config_file), reporthook=report_progress_json)
        logger.info(f"Downloaded .onnx.json file")
        
        # Update status
        with download_lock:
            download_status[model_name]['status'] = DownloadStatus.READY
            download_status[model_name]['progress'] = 100
            download_status[model_name]['downloaded_mb'] = download_status[model_name]['total_mb']
        
        logger.info(f"[{index}/{total}] Download complete: {model_name}")
        
    except Exception as e:
        logger.error(f"[{index}/{total}] Download failed for {model_name}: {e}", exc_info=True)
        with download_lock:
            download_status[model_name]['status'] = DownloadStatus.ERROR
            download_status[model_name]['error'] = str(e)

def get_voice(model_name: str, language: str = None) -> PiperVoice:
    """Get or load voice model by model name (cached, thread-safe)
    
    AGGRESSIVE CACHING: Models are NEVER removed from cache once loaded.
    This prevents freezes during gameplay when switching between NPCs.
    """
    with voice_lock:
        # Check cache first
        if model_name in voice_cache:
            logger.debug(f"Model '{model_name}' found in cache")
            return voice_cache[model_name]
        
        # Model not in cache - need to load it
        logger.info(f"Model '{model_name}' not in cache, loading from disk...")
        
        # Get model info to extract actual filenames from URLs
        if language:
            models = get_all_models_for_language(language)
        else:
            models = get_all_models_for_language(None)  # Get all models
        
        logger.info(f"Looking for model '{model_name}' in language '{language}'")
        logger.info(f"Available models: {list(models.keys())[:10]}...")  # Show first 10
        
        model_info = models.get(model_name)
        
        if model_info:
            # Extract actual filenames from URLs
            url_onnx = model_info.get('url_onnx', '')
            url_json = model_info.get('url_json', '')
            
            onnx_filename = url_onnx.split('/')[-1] if url_onnx else f"{model_name}.onnx"
            json_filename = url_json.split('/')[-1] if url_json else f"{model_name}.onnx.json"
            
            model_file = Path('piper_models') / onnx_filename
            config_file = Path('piper_models') / json_filename
            
            logger.info(f"Loading model '{model_name}' from files: {onnx_filename}, {json_filename}")
        else:
            # Fallback to model_name if not in config
            logger.warning(f"Model '{model_name}' not found in config, using fallback filenames")
            model_file = Path('piper_models') / f"{model_name}.onnx"
            config_file = Path('piper_models') / f"{model_name}.onnx.json"
        
        if not model_file.exists():
            logger.error(f"Model file not found: {model_file}")
            raise FileNotFoundError(f"Model not found: {model_file}")
        
        logger.info(f"Loading model: {model_name} from {model_file}")
        start = time.time()
        
        if config_file.exists():
            voice = PiperVoice.load(str(model_file), config_path=str(config_file))
        else:
            voice = PiperVoice.load(str(model_file))
        
        load_time = time.time() - start
        logger.info(f"Model loaded in {load_time:.2f}s")
        
        # AGGRESSIVE CACHING: Store in cache and NEVER remove
        voice_cache[model_name] = voice
        logger.info(f"Model '{model_name}' cached (total cached: {len(voice_cache)})")
        
        return voice

HERO_PIPER_NAME = "pl_PL-bezimienny-medium"


def native_hero_onnx():
    """Piper onnx trained on Bezimienny. Loaded by the same path as Gosia."""
    root = Path(__file__).parent / "piper_models"
    onnx = root / f"{HERO_PIPER_NAME}.onnx"
    config = root / f"{HERO_PIPER_NAME}.onnx.json"
    if onnx.exists() and config.exists() and onnx.stat().st_size > 1_000_000:
        return HERO_PIPER_NAME
    return None


HERO_RVC_PYTHON = Path(r"C:\Users\mimi\gothic-bezimienny\venv\Scripts\python.exe")
HERO_RVC_SCRIPT = Path(r"C:\Users\mimi\gothic-bezimienny\rvc\hero_rvc_server.py")
HERO_RVC_WEIGHTS = Path(r"C:\Users\mimi\gothic-bezimienny\rvc\assets\weights")
HERO_RVC_URL = "http://127.0.0.1:5556/convert"
_hero_rvc_lock = threading.Lock()


def hero_voice_ids(language):
    with config_lock:
        lang = (config or {}).get("languages", {}).get(language, {})
        return {int(v) for v in (lang.get("hero_voice_ids") or [])}


def hero_weight_ready():
    if not HERO_RVC_WEIGHTS.exists():
        return False
    return any(
        p.stat().st_size > 1000000
        for pattern in ("bezimienny*.pth", "dataset*.pth")
        for p in HERO_RVC_WEIGHTS.glob(pattern)
    )


def ensure_hero_rvc():
    # Converter is off. Bezimienny is trained as a Piper onnx and loaded with the other voices.
    return False
    if not hero_weight_ready() or not HERO_RVC_PYTHON.exists() or not HERO_RVC_SCRIPT.exists():
        return False
    try:
        urllib.request.urlopen("http://127.0.0.1:5556/health", timeout=0.4)
        return True
    except Exception:
        pass
    with _hero_rvc_lock:
        try:
            urllib.request.urlopen("http://127.0.0.1:5556/health", timeout=0.4)
            return True
        except Exception:
            pass
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["MIOPEN_FIND_MODE"] = "1"
        env["MIOPEN_DEBUG_CONV_GEMM"] = "0"
        env["MIOPEN_USER_DB_PATH"] = r"C:\Users\mimi\gothic-bezimienny\rvc\miopen-db"
        env["MIOPEN_CUSTOM_CACHE_DIR"] = r"C:\Users\mimi\gothic-bezimienny\rvc\miopen-db"
        env["HERO_F0_UP"] = "-4"
        env["HERO_INDEX_RATE"] = "0.6"
        if not env.get("USERNAME") and not env.get("USER"):
            env["USERNAME"] = "mimi"
        env["TORCHINDUCTOR_CACHE_DIR"] = r"C:\Users\mimi\gothic-bezimienny\rvc\torch-cache"
        log_path = HERO_RVC_SCRIPT.with_suffix(".log")
        log_handle = open(log_path, "a", encoding="utf-8")
        subprocess.Popen(
            [str(HERO_RVC_PYTHON), str(HERO_RVC_SCRIPT)],
            cwd=str(HERO_RVC_SCRIPT.parent),
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    for _ in range(120):
        time.sleep(0.5)
        try:
            urllib.request.urlopen("http://127.0.0.1:5556/health", timeout=0.4)
            return True
        except Exception:
            continue
    return False


def maybe_convert_hero(audio_array, sample_rate, voice_id, language):
    """Keep Piper's Polish pronunciation and replace the timbre with Bezimienny."""
    if voice_id not in hero_voice_ids(language):
        return None
    if native_hero_onnx():
        return None
    import numpy as np
    if not ensure_hero_rvc():
        logger.warning("Bezimienny RVC is not ready, keeping the Piper carrier voice")
        return None
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(np.ascontiguousarray(audio_array).tobytes())
    request = urllib.request.Request(
        HERO_RVC_URL,
        data=wav_buffer.getvalue(),
        headers={"Content-Type": "audio/wav"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            converted = response.read()
        with wave.open(io.BytesIO(converted), "rb") as wav_file:
            frames = wav_file.readframes(wav_file.getnframes())
            rate = wav_file.getframerate()
        return np.frombuffer(frames, dtype=np.int16).copy(), rate
    except Exception as exc:
        logger.warning(f"Bezimienny conversion failed, keeping the Piper carrier voice: {exc}")
        return None


_VOWELS = set("aąeęioóuy")
_PREFIXES = ("przez", "bez", "roz", "wz", "z")
_WORD = re.compile(r"[A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż]+")
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
_POLISH_STEM = re.compile(
    r"(?i).*(?:ować|ywać|iwać|nąć|ać|eć|ić|yć|"
    r"owan[a-ząćęłńóśźż]*|ując[a-ząćęłńóśźż]*|"
    r"ny|na|ne|ni|ną|nym|nych|nymi|nego|nemu|"
    r"owy|owa|owe|owi|owym|ową|owych|"
    r"ski|ska|skie|ość|ości|"
    r"ł|ła|ło|li|łem|łam|"
    r"uj|yj|my|cie|sz)$"
)


_SYLLABLE = re.compile(r"[aąeęioóuy]+", re.IGNORECASE)


def _mark_trzy(word: str) -> str:
    """Keep the y of trzy when later syllables would take the stress.

    Unstressed, espeak says trzy as czy: trzynastka comes out czynastka.
    A hyphen is the mark espeak actually keeps; an apostrophe does not.
    trzymać stays, because that y is already the stressed syllable.
    """
    lower = word.lower()
    pieces = []
    start = 0
    index = 0
    while index < len(lower):
        if lower.startswith("trzy", index) and index + 4 < len(lower) and lower[index + 4].isalpha():
            if len(_SYLLABLE.findall(lower[index + 4:])) >= 2:
                pieces.append(word[start:index + 4])
                pieces.append("-")
                start = index + 4
                index += 4
                continue
        index += 1
    if not pieces:
        return word
    pieces.append(word[start:])
    return "".join(pieces)


def fix_polish_z(text: str) -> str:
    """Hard z after z-, bez-, roz-, przez-, wz- when the next morpheme starts with i.

    espeak already does that, except the stem starts with im: the letters spell zim,
    so it uses the soft z of zima. The winter root stays soft.
    """
    return _WORD.sub(lambda match: _mark_trzy(_fix_word(match)), text)


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


def generate_tts_worker(req: TTSRequest):
    """Worker thread for TTS generation"""
    try:
        logger.info(f"[{req.request_id}] Starting generation: '{req.text[:50]}...' ({len(req.text)} chars)")
        logger.info(f"[{req.request_id}] Request created, generation started (voice_id={req.voice_id}, gender={req.gender}, length_scale={req.length_scale}, model={req.model_name})")
        logger.info(f"[{req.request_id}] Language: {req.language}, Voice ID: {req.voice_id}")
        
        # Check abort before starting
        if req.should_abort():
            logger.info(f"[{req.request_id}] Aborted before generation")
            return
        
        # Get model name based on language, voice ID, and gender
        model_name = get_model_for_voice(req.language, req.voice_id, req.gender)
        req.model_name = model_name
        logger.info(f"[{req.request_id}] Using model: {model_name}")
        
        # Check if model is already cached
        with voice_lock:
            model_in_cache = model_name in voice_cache
        
        if not model_in_cache:
            # Model needs to be loaded - set status to LOADING_MODEL
            logger.info(f"[{req.request_id}] Model not in cache, loading from disk...")
            req.status = RequestStatus.LOADING_MODEL
        
        # Get voice model (thread-safe, pass language for correct file lookup)
        # This will load from disk if not cached (takes 1-2 seconds)
        voice = get_voice(model_name, req.language)
        
        # Model loaded, back to GENERATING status
        if req.status == RequestStatus.LOADING_MODEL:
            logger.info(f"[{req.request_id}] Model loaded, starting synthesis...")
            req.status = RequestStatus.GENERATING
        
        req.progress = 10
        
        # Check abort before synthesis
        if req.should_abort():
            logger.info(f"[{req.request_id}] Aborted before synthesis")
            return
        
        # Generate audio (thread-safe) with length_scale for speed control
        start = time.time()
        
        with voice_lock:
            # Import SynthesisConfig
            from piper.config import SynthesisConfig
            
            # Create synthesis config with length_scale
            syn_config = SynthesisConfig(
                length_scale=req.length_scale,
                noise_scale=0.667,
                noise_w_scale=0.8
            )
            
            # Collect audio chunks
            audio_chunks = []
            for audio_chunk in voice.synthesize(fix_polish_z(req.text), syn_config=syn_config):
                audio_chunks.append(audio_chunk.audio_int16_bytes)
            
            # Combine all chunks
            audio_data_raw = b''.join(audio_chunks)
        
        # Normalize audio - reduce volume based on user setting before creating WAV
        import numpy as np
        
        # Convert bytes to numpy array (int16)
        audio_array = np.frombuffer(audio_data_raw, dtype=np.int16)
        out_sr = voice.config.sample_rate
        converted = maybe_convert_hero(audio_array, out_sr, req.voice_id, req.language)
        if converted is not None:
            audio_array, out_sr = converted
            logger.info(f"[{req.request_id}] Bezimienny timbre applied")
        
        # Calculate current peak level
        peak = np.max(np.abs(audio_array))
        
        if peak > 0:
            # Volume slider: 0.0 - 1.0 (21 steps in Gothic)
            # 0.0 = -60dB (almost silent)
            # 0.5 = -24dB (default, comfortable)
            # 1.0 = 0dB (full volume, no reduction)
            
            # Convert volume slider (0.0-1.0) to dB reduction
            # Linear mapping: 0.0 -> -60dB, 0.5 -> -24dB, 1.0 -> 0dB
            db_reduction = -60.0 + (req.volume * 60.0)
            
            # Convert dB to linear gain
            # gain = 10^(dB/20)
            gain = 10.0 ** (db_reduction / 20.0)
            
            # Calculate target peak
            target_peak = 32767 * gain
            
            # Apply gain based on peak normalization
            final_gain = target_peak / peak
            
            # Apply gain and clip to prevent overflow
            audio_normalized = np.clip(audio_array * final_gain, -32768, 32767).astype(np.int16)
            
            logger.info(f"[{req.request_id}] Audio normalized: Volume={req.volume:.2f} ({db_reduction:.1f}dB), Peak {peak} -> {int(target_peak)} (gain: {final_gain:.3f}x)")
        else:
            # Silence - no normalization needed
            audio_normalized = audio_array
            logger.warning(f"[{req.request_id}] Audio is silent (peak=0), skipping normalization")
        
        # Create WAV file from normalized audio
        wav_buffer = io.BytesIO()
        import wave
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(out_sr)
            wav_file.writeframes(audio_normalized.tobytes())
        
        synthesis_time = time.time() - start
        wav_buffer.seek(0)
        audio_data = wav_buffer.getvalue()
        
        # Check abort after synthesis
        if req.should_abort():
            logger.info(f"[{req.request_id}] Aborted after synthesis")
            return
        
        # Store result
        req.audio_data = audio_data
        req.status = RequestStatus.READY
        req.progress = 100
        
        logger.info(f"[{req.request_id}] Generated {len(audio_data)} bytes in {synthesis_time:.3f}s ({len(req.text)/synthesis_time:.0f} chars/sec, length_scale={req.length_scale})")
        
    except Exception as e:
        logger.error(f"[{req.request_id}] Error: {e}", exc_info=True)
        req.status = RequestStatus.ERROR
        req.error_message = str(e)

def cleanup_old_requests():
    """Cleanup requests older than 60 seconds"""
    with storage_lock:
        now = datetime.now()
        to_delete = []
        
        for request_id, req in requests_storage.items():
            age = (now - req.created_at).total_seconds()
            if age > 60:
                to_delete.append(request_id)
        
        for request_id in to_delete:
            logger.info(f"[{request_id}] Cleanup: Removing old request")
            del requests_storage[request_id]

@app.route('/health', methods=['GET'])
def health_check():
    """Health check"""
    return jsonify({
        'status': 'ok',
        'service': 'piper-tts-async',
        'server': 'waitress',
        'active_requests': len(requests_storage),
        'cached_models': len(voice_cache)
    })

# Global preload state
preload_state = {
    'is_preloading': False,
    'current_model': '',
    'current_index': 0,
    'total_models': 0
}
preload_state_lock = threading.Lock()

@app.route('/preload_status', methods=['GET'])
def preload_status():
    """Get current preload status
    
    Response:
    {
        "is_preloading": true,
        "current_model": "de_DE-ID03",
        "current_index": 2,
        "total_models": 16
    }
    """
    with preload_state_lock:
        return jsonify(preload_state)

@app.route('/preload_all', methods=['POST', 'GET'])
def preload_all_models():
    """Preload ALL voice models for a language into cache (ASYNC)
    
    Request body (POST):
    {
        "language": "de"
    }
    
    Query parameter (GET):
    ?language=de
    
    Response (immediate):
    {
        "status": "started",
        "message": "Preload started in background",
        "total_models": 19
    }
    """
    try:
        # Support both POST (JSON) and GET (query parameter)
        if request.method == 'POST':
            data = request.get_json()
            language = data.get('language', 'de')
        else:  # GET
            language = request.args.get('language', 'de')
        
        logger.info(f"Preload ALL request: language={language}")
        
        # Check if already preloading
        with preload_state_lock:
            if preload_state['is_preloading']:
                return jsonify({
                    'status': 'already_running',
                    'message': 'Preload already in progress',
                    'current_model': preload_state['current_model'],
                    'current_index': preload_state['current_index'],
                    'total_models': preload_state['total_models']
                })
        
        # Get all unique model files for this language
        models = get_all_models_for_language(language)
        
        # Track unique files (avoid loading duplicates)
        unique_files = {}  # filename -> model_name mapping
        for model_name, model_info in models.items():
            url_onnx = model_info.get('url_onnx', '')
            onnx_filename = url_onnx.split('/')[-1] if url_onnx else f"{model_name}.onnx"
            
            # Only track first occurrence of each file
            if onnx_filename not in unique_files:
                unique_files[onnx_filename] = model_name

        native = native_hero_onnx()
        if native:
            unique_files[f"{native}.onnx"] = native
        
        logger.info(f"Preload ALL: Found {len(unique_files)} unique model files")
        
        # Check which models need to be loaded
        models_to_load = []
        for onnx_filename, model_name in unique_files.items():
            # Check if model file exists
            model_file = Path('piper_models') / onnx_filename
            if not model_file.exists():
                logger.warning(f"Preload ALL: Model file not found: {onnx_filename}")
                continue
            
            # Check if already cached
            with voice_lock:
                if model_name in voice_cache:
                    logger.info(f"Preload ALL: Model '{model_name}' already cached")
                    continue
            
            models_to_load.append((onnx_filename, model_name))
        
        total_models = len(models_to_load)
        logger.info(f"Preload ALL: {total_models} models need loading")
        
        # Start preload in background thread
        def preload_worker():
            try:
                # Set preload state
                with preload_state_lock:
                    preload_state['is_preloading'] = True
                    preload_state['current_model'] = ''
                    preload_state['current_index'] = 0
                    preload_state['total_models'] = total_models
                
                # Load models
                for idx, (onnx_filename, model_name) in enumerate(models_to_load):
                    try:
                        # Update preload state BEFORE loading
                        with preload_state_lock:
                            preload_state['current_model'] = onnx_filename
                            preload_state['current_index'] = idx + 1
                        
                        # Load model (will be cached automatically by get_voice)
                        logger.info(f"Preload ALL: Loading model '{model_name}' from '{onnx_filename}'...")
                        start = time.time()
                        get_voice(model_name, language)
                        load_time = time.time() - start
                        
                        logger.info(f"Preload ALL: Model '{model_name}' loaded in {load_time:.2f}s")
                        
                        # Keep state visible for a moment after loading (so client can see it)
                        time.sleep(0.2)
                        
                    except Exception as e:
                        logger.error(f"Preload ALL: Failed to load {onnx_filename}: {str(e)}")
                
                logger.info(f"Preload ALL complete! Total models in cache: {len(voice_cache)}")
                
            finally:
                # Clear preload state
                with preload_state_lock:
                    preload_state['is_preloading'] = False
                    preload_state['current_model'] = ''
                    preload_state['current_index'] = 0
                    preload_state['total_models'] = 0
        
        # Start background thread
        import threading
        thread = threading.Thread(target=preload_worker, daemon=True)
        thread.start()
        
        # Return immediately
        return jsonify({
            'status': 'started',
            'message': 'Preload started in background',
            'total_models': total_models
        })
        
    except Exception as e:
        logger.error(f"Preload ALL error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/preload', methods=['POST'])
def preload_models():
    """Preload voice models into cache
    
    Request body:
    {
        "language": "de",
        "voice_ids": [1, 3, 4, 5, ...]
    }
    
    Response:
    {
        "status": "success",
        "loaded": 10,
        "failed": 0,
        "models": ["de_DE-ID01", "de_DE-ID03", ...],
        "errors": []
    }
    """
    try:
        data = request.get_json()
        language = data.get('language', 'de')
        voice_ids = data.get('voice_ids', [])
        
        if not voice_ids:
            return jsonify({'error': 'No voice_ids provided'}), 400
        
        logger.info(f"Preload request: language={language}, voice_ids={voice_ids}")
        
        loaded_models = []
        failed_models = []
        errors = []
        
        # First pass: Count how many models need to be loaded
        models_to_load = []
        for voice_id in voice_ids:
            try:
                model_name = get_model_for_voice(language, voice_id, gender=0)
                
                # Check if already cached
                with voice_lock:
                    if model_name in voice_cache:
                        logger.info(f"Preload: Model '{model_name}' already cached (voice_id={voice_id})")
                        loaded_models.append(model_name)
                        continue
                
                models_to_load.append((voice_id, model_name))
            except Exception as e:
                logger.error(f"Preload: Failed to get model for voice_id {voice_id}: {str(e)}")
        
        # Set preload state with actual count of models to load
        with preload_state_lock:
            preload_state['is_preloading'] = True
            preload_state['current_model'] = ''
            preload_state['current_index'] = 0
            preload_state['total_models'] = len(models_to_load)
        
        logger.info(f"Preload: {len(models_to_load)} models need loading, {len(loaded_models)} already cached")
        
        # Second pass: Load models
        for idx, (voice_id, model_name) in enumerate(models_to_load):
            try:
                # Update preload state BEFORE loading
                with preload_state_lock:
                    preload_state['current_model'] = model_name
                    preload_state['current_index'] = idx + 1
                
                # Load model (will be cached automatically by get_voice)
                logger.info(f"Preload: Loading model '{model_name}' (voice_id={voice_id})...")
                start = time.time()
                get_voice(model_name, language)
                load_time = time.time() - start
                
                logger.info(f"Preload: Model '{model_name}' loaded in {load_time:.2f}s")
                loaded_models.append(model_name)
                
                # Keep state visible for a moment after loading (so client can see it)
                time.sleep(0.2)
                
            except Exception as e:
                error_msg = f"Failed to load voice_id {voice_id}: {str(e)}"
                logger.error(f"Preload: {error_msg}")
                failed_models.append(voice_id)
                errors.append(error_msg)
        
        logger.info(f"Preload complete: {len(loaded_models)} loaded, {len(failed_models)} failed")
        logger.info(f"Total models in cache: {len(voice_cache)}")
        
        # Clear preload state
        with preload_state_lock:
            preload_state['is_preloading'] = False
            preload_state['current_model'] = ''
            preload_state['current_index'] = 0
            preload_state['total_models'] = 0
        
        return jsonify({
            'status': 'success' if len(failed_models) == 0 else 'partial',
            'loaded': len(loaded_models),
            'failed': len(failed_models),
            'models': loaded_models,
            'errors': errors,
            'total_cached': len(voice_cache)
        })
        
    except Exception as e:
        logger.error(f"Preload error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/generate', methods=['POST'])
def generate_tts_endpoint():
    """Start TTS generation (async)"""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'Missing text'}), 400
        
        text = data['text']
        language = data.get('language', 'de')
        voice_id = data.get('voice_id')  # Optional voice ID
        gender = data.get('gender', 0)  # Optional gender (0=MALE, 1=FEMALE)
        length_scale = data.get('length_scale', 1.0)  # Optional speed control (default 1.0 = normal)
        volume = data.get('volume', 0.5)  # Optional volume control (default 0.5 = -24dB, 21 steps)
        
        # Convert voice_id to int if provided
        if voice_id is not None:
            try:
                voice_id = int(voice_id)
            except (ValueError, TypeError):
                logger.warning(f"Invalid voice_id: {voice_id}, ignoring")
                voice_id = None
        
        # Validate and clamp length_scale
        try:
            length_scale = float(length_scale)
            length_scale = max(0.5, min(2.0, length_scale))  # Clamp to reasonable range
        except (ValueError, TypeError):
            logger.warning(f"Invalid length_scale: {length_scale}, using default 1.0")
            length_scale = 1.0
        
        # Validate and clamp volume
        try:
            volume = float(volume)
            volume = max(0.0, min(1.0, volume))  # Clamp to 0.0-1.0
        except (ValueError, TypeError):
            logger.warning(f"Invalid volume: {volume}, using default 0.5")
            volume = 0.5
        
        # Get model name for this language/voice_id/gender
        try:
            model_name = get_model_for_voice(language, voice_id, gender)
        except Exception as e:
            logger.error(f"Error getting model for language={language}, voice_id={voice_id}, gender={gender}: {e}")
            return jsonify({
                'error': f'Could not determine model: {str(e)}'
            }), 500
        
        # Check if model exists (extract actual filename from URL)
        models = get_all_models_for_language(language)
        model_info = models.get(model_name)
        
        if model_info:
            # Extract actual filename from URL
            url_onnx = model_info.get('url_onnx', '')
            onnx_filename = url_onnx.split('/')[-1] if url_onnx else f"{model_name}.onnx"
            model_file = Path('piper_models') / onnx_filename
        else:
            # Fallback to model_name
            model_file = Path('piper_models') / f"{model_name}.onnx"
        
        if not model_file.exists():
            logger.error(f"Model not found: {model_file}")
            return jsonify({
                'error': f'Model not found: {model_name}',
                'message': 'Voice model should have been downloaded at server startup'
            }), 500
        
        # Generate request ID
        request_id = str(uuid.uuid4())
        
        # Create request object with gender, length_scale, and volume
        tts_request = TTSRequest(request_id, text, language, voice_id, gender, length_scale, volume)
        
        # Store request
        with storage_lock:
            requests_storage[request_id] = tts_request
        
        # Start worker thread
        worker = threading.Thread(target=generate_tts_worker, args=(tts_request,), daemon=True)
        worker.start()
        
        logger.info(f"[{request_id}] Request created, generation started (voice_id={voice_id}, gender={gender}, length_scale={length_scale}, volume={volume}, model={model_name})")
        
        # Cleanup old requests
        cleanup_old_requests()
        
        return jsonify({
            'request_id': request_id,
            'status': 'generating',
            'model': model_name
        })
        
    except Exception as e:
        logger.error(f"Error in /generate: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/status/<request_id>', methods=['GET'])
def get_status(request_id):
    """Get generation status"""
    with storage_lock:
        req = requests_storage.get(request_id)
    
    if not req:
        return jsonify({'error': 'Request not found'}), 404
    
    return jsonify({
        'request_id': request_id,
        'status': req.status,
        'progress': req.progress,
        'error': req.error_message,
        'model': req.model_name  # NEW: Include model name for UI display
    })

@app.route('/audio/<request_id>', methods=['GET'])
def get_audio(request_id):
    """Get generated audio"""
    with storage_lock:
        req = requests_storage.get(request_id)
    
    if not req:
        return jsonify({'error': 'Request not found'}), 404
    
    if req.status != RequestStatus.READY:
        return jsonify({
            'error': 'Audio not ready',
            'status': req.status
        }), 400
    
    if not req.audio_data:
        return jsonify({'error': 'No audio data'}), 500
    
    logger.info(f"[{request_id}] Sending audio ({len(req.audio_data)} bytes)")
    
    return send_file(
        io.BytesIO(req.audio_data),
        mimetype='audio/wav',
        as_attachment=False
    )

@app.route('/abort/<request_id>', methods=['POST'])
def abort_generation(request_id):
    """Abort TTS generation"""
    with storage_lock:
        req = requests_storage.get(request_id)
    
    if not req:
        return jsonify({'error': 'Request not found'}), 404
    
    if req.status == RequestStatus.GENERATING:
        req.abort()
        logger.info(f"[{request_id}] Abort requested")
        return jsonify({
            'request_id': request_id,
            'status': 'aborted'
        })
    else:
        return jsonify({
            'request_id': request_id,
            'status': req.status,
            'message': 'Request not generating'
        })

@app.route('/download_status', methods=['GET'])
def get_download_status_all():
    """Get download status for all models"""
    logger.info(f"[DEBUG] /download_status called - download_status has {len(download_status)} entries")
    logger.info(f"[DEBUG] /download_status called - total_downloads = {total_downloads}")
    
    with download_lock:
        # Build response with all models
        models_status = []
        
        logger.info(f"[DEBUG] Inside lock - download_status has {len(download_status)} entries")
        
        for model_name, status in download_status.items():
            models_status.append({
                'model': model_name,
                'status': status['status'],
                'progress': status['progress'],
                'total_mb': status.get('total_mb', 0),
                'downloaded_mb': status.get('downloaded_mb', 0),
                'error': status.get('error'),
                'index': status.get('index', 0),
                'total': status.get('total', 0)
            })
        
        logger.info(f"[DEBUG] Built models_status with {len(models_status)} entries")
        
        # Sort by index
        models_status.sort(key=lambda x: x['index'])
        
        # Calculate overall progress
        if models_status:
            total_progress = sum(m['progress'] for m in models_status)
            overall_progress = total_progress // len(models_status)
        else:
            overall_progress = 100
        
        logger.info(f"[DEBUG] Returning: models={len(models_status)}, overall_progress={overall_progress}, total_models={total_downloads}")
        
        return jsonify({
            'models': models_status,
            'overall_progress': overall_progress,
            'current_index': current_download_index,
            'total_models': total_downloads
        })

@app.route('/download_status/<language>', methods=['GET'])
def get_download_status(language):
    """Get download status for a language (legacy endpoint)"""
    # This endpoint is kept for backwards compatibility
    # It returns the status of the default model for the language
    
    try:
        default_model = get_model_for_voice(language, None)
    except:
        return jsonify({
            'language': language,
            'status': 'error',
            'error': 'Language not found in configuration'
        }), 404
    
    with download_lock:
        if default_model not in download_status:
            # Check if model already exists
            model_file = Path('piper_models') / f"{default_model}.onnx"
            if model_file.exists():
                return jsonify({
                    'language': language,
                    'model': default_model,
                    'status': 'ready',
                    'progress': 100
                })
            else:
                return jsonify({
                    'language': language,
                    'model': default_model,
                    'status': 'idle',
                    'progress': 0
                })
        
        status = download_status[default_model]
        return jsonify({
            'language': language,
            'model': default_model,
            'status': status['status'],
            'progress': status['progress'],
            'total_mb': status.get('total_mb', 0),
            'downloaded_mb': status.get('downloaded_mb', 0),
            'error': status.get('error')
        })

def check_and_download_models(language: str):
    """Check and download all required voice models for a language"""
    global download_queue, total_downloads
    
    logger.info("=" * 60)
    logger.info(f"Checking voice models for language: {language}")
    
    # Get all models for this language
    models = get_all_models_for_language(language)
    
    if not models:
        logger.error(f"No models defined for language: {language}")
        return False
    
    logger.info(f"Found {len(models)} model(s) defined for language '{language}'")
    
    # Check which models need to be downloaded
    # Track unique files to avoid downloading duplicates
    models_to_download = []
    seen_files = set()  # Track (onnx_filename, json_filename) tuples
    
    for model_name, model_info in models.items():
        # Extract actual filenames from URLs (they might differ from model_name)
        url_onnx = model_info.get('url_onnx', '')
        url_json = model_info.get('url_json', '')
        
        # Extract filename from URL (e.g., "de_DE-ID01.onnx" from URL)
        onnx_filename = url_onnx.split('/')[-1] if url_onnx else f"{model_name}.onnx"
        json_filename = url_json.split('/')[-1] if url_json else f"{model_name}.onnx.json"
        
        # Check if files exist with the actual filenames from URLs
        model_file = Path('piper_models') / onnx_filename
        config_file = Path('piper_models') / json_filename
        
        if model_file.exists() and config_file.exists():
            size_mb = model_file.stat().st_size / 1024 / 1024
            logger.info(f"✓ Model '{model_name}' already exists as '{onnx_filename}' ({size_mb:.1f} MB)")
        else:
            # Check if we already queued this file for download
            file_key = (onnx_filename, json_filename)
            
            if file_key not in seen_files:
                logger.info(f"✗ Model '{model_name}' missing (expected: '{onnx_filename}') - will download")
                models_to_download.append((model_name, model_info))
                seen_files.add(file_key)
            else:
                logger.info(f"⊙ Model '{model_name}' points to '{onnx_filename}' (already queued for download)")
    
    if not models_to_download:
        logger.info("All models already downloaded!")
        logger.info("=" * 60)
        return True
    
    # Download missing models sequentially
    logger.info(f"Downloading {len(models_to_download)} model(s)...")
    logger.info("=" * 60)
    
    download_queue = models_to_download
    total_downloads = len(models_to_download)
    
    for index, (model_name, model_info) in enumerate(models_to_download, start=1):
        logger.info(f"Starting download {index}/{total_downloads}: {model_name}")
        download_voice_model(model_name, model_info, index, total_downloads)
        
        # Wait for download to complete
        while True:
            with download_lock:
                status = download_status.get(model_name, {})
                current_status = status.get('status', DownloadStatus.IDLE)
            
            if current_status == DownloadStatus.READY:
                logger.info(f"✓ Download {index}/{total_downloads} complete: {model_name}")
                break
            elif current_status == DownloadStatus.ERROR:
                error = status.get('error', 'Unknown error')
                logger.error(f"✗ Download {index}/{total_downloads} failed: {model_name}")
                logger.error(f"  Error: {error}")
                logger.info("=" * 60)
                return False
            
            time.sleep(0.5)
    
    logger.info("=" * 60)
    logger.info("All downloads complete!")
    logger.info("=" * 60)
    return True

def init_download_status_for_language(language: str):
    """Initialize download status for all models that need downloading
    
    This function MUST be called BEFORE the server starts to ensure
    /download_status endpoint returns correct data immediately.
    """
    global download_status, download_queue, total_downloads
    
    models = get_all_models_for_language(language)
    models_to_download = []
    seen_files = set()  # Track unique files to avoid duplicates
    
    for model_name, model_info in models.items():
        # Extract actual filenames from URLs (they might differ from model_name)
        url_onnx = model_info.get('url_onnx', '')
        url_json = model_info.get('url_json', '')
        
        # Extract filename from URL (e.g., "de_DE-ID01.onnx" from URL)
        onnx_filename = url_onnx.split('/')[-1] if url_onnx else f"{model_name}.onnx"
        json_filename = url_json.split('/')[-1] if url_json else f"{model_name}.onnx.json"
        
        # Check if files exist with the actual filenames from URLs
        model_file = Path('piper_models') / onnx_filename
        config_file = Path('piper_models') / json_filename
        
        logger.info(f"Checking model '{model_name}':")
        logger.info(f"  Expected .onnx: {model_file}")
        logger.info(f"  Expected .json: {config_file}")
        logger.info(f"  .onnx exists: {model_file.exists()}")
        logger.info(f"  .json exists: {config_file.exists()}")
        
        if not (model_file.exists() and config_file.exists()):
            # Check if we already queued this file for download
            file_key = (onnx_filename, json_filename)
            
            if file_key not in seen_files:
                models_to_download.append((model_name, model_info))
                seen_files.add(file_key)
                
                # Initialize status immediately
                with download_lock:
                    download_status[model_name] = {
                        'status': DownloadStatus.IDLE,
                        'progress': 0,
                        'total_mb': model_info.get('size_mb', 50),
                        'downloaded_mb': 0,
                        'error': None,
                        'index': len(download_status) + 1,
                        'total': 0
                    }
            else:
                logger.info(f"  ⊙ Model '{model_name}' points to '{onnx_filename}' (already queued)")
        else:
            size_mb = model_file.stat().st_size / 1024 / 1024
            logger.info(f"  ✓ Model '{model_name}' already exists, skipping download")
            with download_lock:
                download_status[model_name] = {
                    'status': DownloadStatus.READY,
                    'progress': 100,
                    'total_mb': round(size_mb, 1),
                    'downloaded_mb': round(size_mb, 1),
                    'error': None,
                    'index': len(download_status) + 1,
                    'total': 0
                }
    
    # Missing files are what still needs a download. The status list also
    # includes models already on disk, so the game can see them as ready.
    missing_count = len(models_to_download)
    reported_count = len(download_status)
    with download_lock:
        for model_name in download_status:
            download_status[model_name]['total'] = reported_count
    
    download_queue = models_to_download
    total_downloads = reported_count
    total_count = missing_count
    
    logger.info(f"Download status initialized: {len(download_status)} entries, {total_downloads} total downloads")
    logger.info(f"Global download_status has {len(download_status)} entries")
    logger.info(f"Global total_downloads = {total_downloads}")
    
    return total_count  # Return the number of downloads needed

def main():
    parser = argparse.ArgumentParser(description='Piper-TTS Server (Async)')
    parser.add_argument('--port', type=int, default=5555, help='Server port')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Server host')
    parser.add_argument('--threads', type=int, default=4, help='Worker threads')
    parser.add_argument('--language', type=str, default='de', help='Language to use (default: de)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode (verbose logging)')
    args = parser.parse_args()
    
    # Set logging level based on debug flag
    if not args.debug:
        # Reduce logging to WARNING level (only errors and warnings)
        logging.getLogger().setLevel(logging.WARNING)
        logger.setLevel(logging.WARNING)
    
    logger.info("=" * 60)
    logger.info("Starting Piper-TTS Server (ASYNC)")
    logger.info("=" * 60)
    logger.info(f"Server: Waitress (production WSGI)")
    logger.info(f"Threads: {args.threads}")
    logger.info(f"Address: {args.host}:{args.port}")
    logger.info(f"Language: {args.language}")
    logger.info(f"Debug Mode: {args.debug}")
    logger.info("=" * 60)
    
    # Load configuration from YAML
    try:
        load_config()
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        logger.error("Server cannot start without configuration file")
        input("Press Enter to exit...")
        sys.exit(1)
    
    # Initialize download status BEFORE starting server
    # This ensures /download_status endpoint works immediately
    logger.info("Initializing download status...")
    num_downloads = init_download_status_for_language(args.language)
    downloads_needed = num_downloads > 0
    
    logger.info(f"After init: downloads_needed={downloads_needed}, num_downloads={num_downloads}")
    logger.info(f"After init: global total_downloads={total_downloads}")
    logger.info(f"After init: global download_status has {len(download_status)} entries")
    
    # Start download in background thread
    def download_and_warmup():
        if downloads_needed:
            logger.info(f"Starting background download of {total_downloads} model(s)...")
            if not check_and_download_models(args.language):
                logger.error("Failed to download voice models!")
                return
        
        # Warmup: Pre-load default voice model
        try:
            default_model = get_model_for_voice(args.language, None)
            logger.info(f"Warming up default voice model: {default_model}")
            warmup_start = time.time()
            
            voice = get_voice(default_model, args.language)
            # Generate a short test audio
            test_text = "Hallo"
            wav_buffer = io.BytesIO()
            import wave
            with wave.open(wav_buffer, 'wb') as wav_file:
                voice.synthesize_wav(test_text, wav_file)
            warmup_time = time.time() - warmup_start
            logger.info(f"Warmup complete in {warmup_time:.2f}s - Voice model ready!")
        except Exception as e:
            logger.warning(f"Warmup failed: {e}")
            logger.warning("Server will continue, but first request may be slower")
    
    # Start download in background
    download_thread = threading.Thread(target=download_and_warmup, daemon=True)
    download_thread.start()
    
    # Start server immediately (download status is already initialized)
    logger.info("Starting Flask server...")
    if downloads_needed:
        logger.info(f"Note: {total_downloads} model(s) downloading in background")
        logger.info("Client can query /download_status for progress")
    
    try:
        from waitress import serve
        logger.info("Using Waitress server (async)")
        
        # Reduce Waitress logging if not in debug mode
        if not args.debug:
            import logging as waitress_logging
            waitress_logging.getLogger('waitress').setLevel(waitress_logging.WARNING)
        
        serve(app, host=args.host, port=args.port, threads=args.threads)
    except ImportError:
        logger.warning("Waitress not installed, falling back to Flask dev server")
        logger.warning("Install for better performance: pip install waitress")
        app.run(host=args.host, port=args.port, debug=False, threaded=True)

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        input("Press Enter to exit...")
        sys.exit(1)
