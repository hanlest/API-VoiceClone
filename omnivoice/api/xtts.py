"""Endpoint de prueba con XTTS-v2.

El audio de referencia solo aporta el timbre. El idioma de ese clip no se pide:
``language`` es el idioma del texto que se va a generar.
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import tempfile
import threading
from pathlib import Path
from typing import Literal

import numpy as np
import soundfile as sf
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel

from omnivoice.utils.common import get_best_device

logger = logging.getLogger("omnivoice.api.xtts")

MAX_AUDIO_BYTES = 25 * 1024 * 1024
MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
_AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

# Códigos oficiales de XTTS-v2 y alias de los nombres que ya usa esta API.
_LANGUAGES = {
    "en": "en",
    "english": "en",
    "es": "es",
    "spanish": "es",
    "español": "es",
    "espanol": "es",
    "fr": "fr",
    "french": "fr",
    "de": "de",
    "german": "de",
    "it": "it",
    "italian": "it",
    "pt": "pt",
    "portuguese": "pt",
    "pl": "pl",
    "polish": "pl",
    "tr": "tr",
    "turkish": "tr",
    "ru": "ru",
    "russian": "ru",
    "nl": "nl",
    "dutch": "nl",
    "cs": "cs",
    "czech": "cs",
    "ar": "ar",
    "arabic": "ar",
    "zh-cn": "zh-cn",
    "zh": "zh-cn",
    "chinese": "zh-cn",
    "ja": "ja",
    "japanese": "ja",
    "hu": "hu",
    "hungarian": "hu",
    "ko": "ko",
    "korean": "ko",
    "hi": "hi",
    "hindi": "hi",
}

router = APIRouter(prefix="/v1/xtts", tags=["XTTS"])


class XttsModelLoadResponse(BaseModel):
    status: str
    model: str
    device: str
    sampling_rate: int | None = None
    already_loaded: bool
    vram_reserved_mib: float | None = None


_model = None
_model_lock = threading.Lock()
_infer_lock = threading.Lock()


def _output_language(language: str) -> str:
    code = _LANGUAGES.get(language.strip().lower())
    if code is None:
        supported = "en, es, fr, de, it, pt, pl, tr, ru, nl, cs, ar, zh-cn, ja, hu, ko, hi"
        raise HTTPException(
            status_code=400,
            detail=f"Idioma de salida no admitido: {language}. Usa uno de: {supported}.",
        )
    return code


def _prepare_tts_import() -> None:
    """coqui-tts 0.27 sigue importando un helper que Transformers 5 ya no exporta."""
    import transformers.pytorch_utils as pytorch_utils

    if hasattr(pytorch_utils, "isin_mps_friendly"):
        return
    import torch

    pytorch_utils.isin_mps_friendly = torch.isin


def _load_model():
    global _model
    if _model is not None:
        return _model
    with _model_lock:
        if _model is not None:
            return _model
        os.environ["COQUI_TOS_AGREED"] = "1"
        _prepare_tts_import()
        try:
            from TTS.api import TTS
        except ImportError as exc:
            raise HTTPException(
                status_code=503,
                detail="Falta coqui-tts. Instálalo con: pip install coqui-tts",
            ) from exc
        device = get_best_device()
        logger.info("Cargando %s en %s", MODEL_NAME, device)
        try:
            _model = TTS(MODEL_NAME).to(device)
        except Exception as exc:
            logger.exception("No se pudo cargar XTTS-v2")
            raise HTTPException(
                status_code=500,
                detail=f"No se pudo cargar XTTS-v2: {type(exc).__name__}: {exc}",
            ) from exc
        return _model


def _mount_xtts() -> bool:
    with _model_lock:
        already_loaded = _model is not None
    if already_loaded:
        return True
    _load_model()
    return False


def _xtts_device() -> str:
    model = _model
    if model is None:
        return get_best_device()
    try:
        return str(next(model.parameters()).device)
    except StopIteration:
        return get_best_device()


def _synthesize(
    text: str,
    language: str,
    speaker_wav: str,
    settings: dict,
) -> tuple[np.ndarray, int]:
    model = _load_model()
    try:
        with _infer_lock:
            wav = model.tts(
                text=text,
                speaker_wav=speaker_wav,
                language=language,
                **settings,
            )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Error al sintetizar con XTTS-v2")
        raise HTTPException(
            status_code=500,
            detail=f"{type(exc).__name__}: {exc}",
        ) from exc

    audio = np.asarray(wav, dtype=np.float32).reshape(-1)
    sample_rate = int(getattr(model.synthesizer, "output_sample_rate", 24000))
    return audio, sample_rate


def _unlink(path: str) -> None:
    try:
        Path(path).unlink(missing_ok=True)
    except OSError:
        logger.warning("No se pudo borrar el audio temporal %s", path)


def _trim_wav(path: str, *, mid_sil: int, lead_sil: int, trail_sil: int) -> str:
    from omnivoice.utils.audio import remove_silence

    data, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    audio = data.T
    if audio.shape[0] > 1:
        audio = np.mean(audio, axis=0, keepdims=True)
    trimmed = remove_silence(
        audio,
        sample_rate,
        mid_sil=mid_sil,
        lead_sil=lead_sil,
        trail_sil=trail_sil,
    )
    if trimmed.size == 0:
        raise HTTPException(
            status_code=400,
            detail="El audio quedó vacío al quitar los silencios.",
        )
    fd, dest = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    wave = trimmed[0] if trimmed.ndim == 2 else trimmed
    sf.write(dest, np.asarray(wave, dtype=np.float32), sample_rate, format="WAV")
    return dest


def _trim_generated(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    from omnivoice.utils.audio import remove_silence

    trimmed = remove_silence(
        audio.reshape(1, -1),
        sample_rate,
        mid_sil=500,
        lead_sil=100,
        trail_sil=100,
    )
    if trimmed.size == 0:
        return audio
    return np.asarray(trimmed, dtype=np.float32).reshape(-1)


def _normalize_output_text(text: str, language: str) -> str:
    from omnivoice.utils.text import normalize_text

    code = "zh" if language == "zh-cn" else language
    try:
        normalized = normalize_text(text, code)
    except ImportError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not normalized.strip():
        raise HTTPException(
            status_code=400,
            detail="El texto quedó vacío después de normalizarlo.",
        )
    return normalized


def _encode_waveform(audio: np.ndarray, sample_rate: int, audio_format: str) -> tuple[bytes, int]:
    from omnivoice.api.app import _AUDIO_FORMATS

    spec = _AUDIO_FORMATS[audio_format]
    buffer = io.BytesIO()
    try:
        sf.write(
            buffer,
            np.asarray(audio, dtype=np.float32).reshape(-1),
            sample_rate,
            format=spec["soundfile"],
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"No se pudo escribir el audio en {audio_format}: {exc}",
        ) from exc
    return buffer.getvalue(), sample_rate


def _make_produce(speaker_path: str, form: dict, delete_speaker: bool):
    def produce(on_progress=None):
        del on_progress
        trimmed = None
        try:
            text = form["text"]
            if form["normalize_text"]:
                text = _normalize_output_text(text, form["language"])
            speaker = speaker_path
            if form["preprocess_prompt"]:
                trimmed = _trim_wav(speaker_path, mid_sil=200, lead_sil=100, trail_sil=200)
                speaker = trimmed
            audio, sample_rate = _synthesize(
                text, form["language"], speaker, form["settings"]
            )
            if form["postprocess_output"]:
                audio = _trim_generated(audio, sample_rate)
            return _encode_waveform(audio, sample_rate, form["audio_format"])
        finally:
            if trimmed:
                _unlink(trimmed)
            if delete_speaker:
                _unlink(speaker_path)

    return produce


async def _respond(request: Request, speaker_path: str, form: dict, *, delete_speaker: bool):
    from omnivoice.api.app import DeliveryOptions, _deliver_produced

    delivery = DeliveryOptions(
        generation_mode=form["generation_mode"],
        audio_response=form["audio_response"],
        audio_format=form["audio_format"],
    )
    return await _deliver_produced(
        request,
        _make_produce(speaker_path, form, delete_speaker),
        "xtts",
        delivery,
    )


def _generation_settings(
    temperature: float,
    repetition_penalty: float,
    top_k: int,
    top_p: float,
    num_beams: int,
    gpt_cond_len: int,
    speed: float,
) -> dict:
    settings = {
        "temperature": temperature,
        "repetition_penalty": repetition_penalty,
        "top_k": top_k,
        "top_p": top_p,
        "num_beams": num_beams,
        "gpt_cond_len": gpt_cond_len,
        "speed": speed,
    }
    if num_beams > 1:
        settings["do_sample"] = False
    if gpt_cond_len < 4:
        settings["gpt_cond_chunk_len"] = gpt_cond_len
    return settings


def _xtts_form(
    text: str = Form(..., description="Texto a sintetizar en el idioma de salida."),
    language: str = Form(
        "es",
        description=(
            "Idioma del audio generado: en, es, fr, de, it, pt, pl, tr, ru, nl, cs, "
            "ar, zh-cn, ja, hu, ko o hi. También acepta English, Spanish, French, "
            "Chinese o Japanese. El idioma del audio de referencia no se indica."
        ),
    ),
    temperature: float = Form(
        0.75,
        ge=0.05,
        le=1.5,
        description="0,75 es el valor del modelo. Bajar hacia 0,65 estabiliza la pronunciación. Subir la hace más expresiva.",
    ),
    repetition_penalty: float = Form(
        5.0,
        ge=1.0,
        le=20.0,
        description="5 es el valor del modelo. Subir hacia 10 reduce repeticiones, tartamudeos y silencios largos.",
    ),
    top_k: int = Form(
        50,
        ge=1,
        le=100,
        description="50 es el valor del modelo. Un valor menor hace la voz más predecible.",
    ),
    top_p: float = Form(
        0.85,
        gt=0.0,
        le=1.0,
        description="0,85 es el valor del modelo. Un valor menor hace la voz más predecible.",
    ),
    num_beams: int = Form(
        1,
        ge=1,
        le=4,
        description=(
            "1 genera una sola candidata. 2 o 4 exploran más frases y tardan más. "
            "A partir de 2 se usa búsqueda por haces: temperatura, top_k y top_p dejan de aplicarse."
        ),
    ),
    gpt_cond_len: int = Form(
        30,
        ge=3,
        le=30,
        description="Segundos del audio de referencia usados para el timbre. 30 es el máximo del modelo.",
    ),
    speed: float = Form(
        1.0,
        ge=0.5,
        le=2.0,
        description="1 es el ritmo normal. Un valor menor alarga el audio ya generado y uno mayor lo acorta.",
    ),
    preprocess_prompt: bool = Form(
        True,
        description="Recorta silencios del audio de referencia antes de clonar.",
    ),
    postprocess_output: bool = Form(
        True,
        description="Quita silencios largos del audio generado.",
    ),
    normalize_text: bool = Form(
        False,
        description=(
            "Convierte números a su forma hablada. En inglés y chino también fechas y montos. "
            "En los demás idiomas, solo los enteros."
        ),
    ),
    generation_mode: Literal["sync", "sse", "job"] = Form(
        "sync",
        description=(
            "sync espera el audio en esta petición. sse envía el progreso por Server-Sent Events. "
            "job responde al momento; el progreso se consulta en el estado del trabajo. "
            "XTTS no publica pasos intermedios."
        ),
    ),
    audio_response: Literal["base64", "archivo", "url"] = Form(
        "archivo",
        description=(
            "base64 devuelve el audio codificado. archivo lo descarga. url devuelve un enlace. "
            "En sse y job, archivo también se entrega como enlace de descarga."
        ),
    ),
    audio_format: Literal["wav", "flac", "ogg", "mp3", "aiff"] = Form(
        "wav",
        description="Formato del audio: wav, flac, ogg, mp3 o aiff.",
    ),
) -> dict:
    cleaned = text.strip()
    if not cleaned:
        raise HTTPException(status_code=400, detail="El texto a sintetizar está vacío.")
    return {
        "text": cleaned,
        "language": _output_language(language),
        "settings": _generation_settings(
            temperature,
            repetition_penalty,
            top_k,
            top_p,
            num_beams,
            gpt_cond_len,
            speed,
        ),
        "preprocess_prompt": preprocess_prompt,
        "postprocess_output": postprocess_output,
        "normalize_text": normalize_text,
        "generation_mode": generation_mode,
        "audio_response": audio_response,
        "audio_format": audio_format,
    }


_SYNTHESIS_RESPONSES = {
    200: {
        "description": "Resultado según generation_mode y audio_response.",
        "content": {
            "audio/wav": {"schema": {"type": "string", "format": "binary"}},
            "application/json": {"schema": {"type": "object"}},
            "text/event-stream": {"schema": {"type": "string"}},
        },
    },
    202: {
        "description": "Trabajo encolado cuando generation_mode es job.",
        "content": {"application/json": {"schema": {"type": "object"}}},
    },
}


def _saved_reference_path(voice_id: str) -> Path:
    from omnivoice.api.app import REFERENCE_CLIP_NAME, _read_voice, _resolve_voice_id, _voice_dir

    voice_id = _resolve_voice_id(voice_id)
    folder = _voice_dir(voice_id)
    if _read_voice(folder) is None:
        raise HTTPException(status_code=404, detail="Voz no encontrada.")
    clip = folder / REFERENCE_CLIP_NAME
    if not clip.is_file():
        raise HTTPException(
            status_code=400,
            detail=(
                "Esta voz se guardó sin el audio original. "
                "Vuelve a crearla para usarla con XTTS."
            ),
        )
    return clip


@router.post(
    "/model/load",
    response_model=XttsModelLoadResponse,
    summary="Montar XTTS-v2 en VRAM",
)
async def load_xtts_model():
    """Carga el checkpoint de XTTS-v2 sin sintetizar audio."""
    from omnivoice.api.app import _cuda_reserved_mib

    try:
        already_loaded = await asyncio.to_thread(_mount_xtts)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("No se pudo cargar XTTS-v2")
        raise HTTPException(
            status_code=500,
            detail=f"No se pudo cargar XTTS-v2: {type(exc).__name__}: {exc}",
        ) from exc

    model = _model
    sample_rate = None
    if model is not None:
        sample_rate = int(getattr(model.synthesizer, "output_sample_rate", 24000))

    return XttsModelLoadResponse(
        status="ok",
        model=MODEL_NAME,
        device=_xtts_device(),
        sampling_rate=sample_rate,
        already_loaded=already_loaded,
        vram_reserved_mib=_cuda_reserved_mib(),
    )


@router.post(
    "/synthesize",
    summary="Clonar un audio y hablar en otro idioma",
    responses=_SYNTHESIS_RESPONSES,
)
async def synthesize(
    request: Request,
    reference_audio: UploadFile = File(
        ...,
        description="Audio original de la voz. Unos 6 segundos bastan. Puede estar en otro idioma.",
    ),
    form: dict = Depends(_xtts_form),
):
    """Genera voz con XTTS-v2 a partir de un clip y un texto.

    El clip solo define el timbre. ``language`` es el idioma en el que se lee el texto.
    """
    raw = await reference_audio.read()
    if not raw:
        raise HTTPException(status_code=400, detail="El audio de referencia está vacío.")
    if len(raw) > MAX_AUDIO_BYTES:
        raise HTTPException(
            status_code=413,
            detail="El audio de referencia supera los 25 MB.",
        )

    suffix = Path(reference_audio.filename or "ref.wav").suffix.lower()
    if suffix not in _AUDIO_SUFFIXES:
        suffix = ".wav"

    fd, speaker_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    try:
        Path(speaker_path).write_bytes(raw)
    except Exception:
        _unlink(speaker_path)
        raise
    return await _respond(request, speaker_path, form, delete_speaker=True)


@router.post(
    "/voices/{voice_id}/synthesize",
    summary="Hablar en otro idioma con una voz guardada",
    responses=_SYNTHESIS_RESPONSES,
)
async def synthesize_saved_voice(
    voice_id: str,
    request: Request,
    form: dict = Depends(_xtts_form),
):
    """Usa el clip original guardado con la voz y sintetiza el texto con XTTS-v2."""
    speaker_path = _saved_reference_path(voice_id)
    return await _respond(request, str(speaker_path), form, delete_speaker=False)
