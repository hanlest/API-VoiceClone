"""API FastAPI de OmniVoice.

Swagger UI: ``/docs``
ReDoc: ``/redoc``
Esquema OpenAPI: ``/openapi.json``
"""

from __future__ import annotations

import asyncio
import base64
import gc
import io
import json
import logging
import os
import queue
import re
import shutil
import threading
import time
import uuid
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
import soundfile as sf
import torch
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.openapi.utils import get_openapi
from fastapi.responses import (
    FileResponse,
    JSONResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)
from pydantic import BaseModel, BeforeValidator, Field

from omnivoice import OmniVoice, OmniVoiceGenerationConfig, VoiceClonePrompt, __version__
from omnivoice.utils.audio import load_audio_bytes
from omnivoice.utils.common import get_best_device
from omnivoice.utils.lang_map import LANG_NAMES, lang_display_name

logger = logging.getLogger("omnivoice.api")

MAX_AUDIO_BYTES = 25 * 1024 * 1024

Gender = Literal["male", "female"]
Age = Literal["child", "teenager", "young adult", "middle-aged", "elderly"]
Pitch = Literal[
    "very low pitch",
    "low pitch",
    "moderate pitch",
    "high pitch",
    "very high pitch",
]
GenerationMode = Literal["sync", "sse", "job"]
AudioResponseMode = Literal["base64", "archivo", "url"]
AudioFormat = Literal["wav", "flac", "ogg", "mp3", "aiff"]
MainLanguage = Literal["English", "Chinese", "Japanese", "Spanish", "French"]
_AUDIO_FORMATS = {
    "wav": {"soundfile": "WAV", "extension": "wav", "media_type": "audio/wav"},
    "flac": {"soundfile": "FLAC", "extension": "flac", "media_type": "audio/flac"},
    "ogg": {"soundfile": "OGG", "extension": "ogg", "media_type": "audio/ogg"},
    "mp3": {"soundfile": "MP3", "extension": "mp3", "media_type": "audio/mpeg"},
    "aiff": {"soundfile": "AIFF", "extension": "aiff", "media_type": "audio/aiff"},
}

_infer_lock = threading.Lock()
_jobs_lock = threading.Lock()
_model_lock = threading.Lock()
_model_users = 0
_last_used = time.monotonic()
_UNLOAD_POLL_SECONDS = 5.0
_jobs: dict[str, dict] = {}
_AUDIO_DIR = Path("outputs") / "audio"
_VOICES_DIR = Path("voices")
_AUDIO_NAME_RE = re.compile(r"^[0-9a-f]{32}\.(wav|flac|ogg|mp3|aiff)$")
_VOICE_ID_RE = re.compile(r"^[0-9a-f]{32}$")


class Settings(BaseModel):
    model: str = "k2-fsa/OmniVoice"
    device: str | None = None
    no_asr: bool = False
    asr_model: str = "openai/whisper-large-v3-turbo"
    idle_unload_seconds: float = 60


class HealthResponse(BaseModel):
    status: str
    model: str
    device: str
    sampling_rate: int | None = None
    asr_enabled: bool
    model_loaded: bool


class LanguageList(BaseModel):
    languages: list[str]


class SavedVoice(BaseModel):
    id: str
    name: str
    created_at: str
    reference_text: str
    language: MainLanguage | None = None


class SavedVoiceList(BaseModel):
    voices: list[SavedVoice]


class DeliveryOptions(BaseModel):
    generation_mode: GenerationMode = "sync"
    audio_response: AudioResponseMode = "archivo"
    audio_format: AudioFormat = "wav"


class GenerationOptions(BaseModel):
    speed: float = Field(
        1.0,
        ge=0.5,
        le=1.5,
        description="1.0 es el ritmo normal. Un valor mayor acelera y uno menor ralentiza. Se ignora si duration está definido.",
    )
    duration: float | None = Field(
        None,
        gt=0,
        description="Duración fija en segundos. Si se define, reemplaza a speed.",
    )
    num_step: int = Field(
        32,
        ge=4,
        le=64,
        description="Pasos de inferencia. Menos pasos es más rápido; más pasos suele mejorar la calidad.",
    )
    guidance_scale: float = Field(
        2.0,
        ge=0.0,
        le=4.0,
        description="Escala de classifier-free guidance.",
    )
    denoise: bool = Field(True, description="Activa el token de reducción de ruido.")
    preprocess_prompt: bool = Field(
        True,
        description="Recorta silencios del audio de referencia y ajusta la puntuación del texto de referencia.",
    )
    postprocess_output: bool = Field(
        True,
        description="Quita silencios largos del audio generado.",
    )
    normalize_text: bool = Field(
        False,
        description="Convierte números, fechas y montos a su forma hablada.",
    )


def _blank_to_none(value):
    if isinstance(value, str) and not value.strip():
        return None
    return value


_Blank = BeforeValidator(_blank_to_none)
OptionalStr = Annotated[str | None, _Blank]
OptionalFloat = Annotated[float | None, _Blank]
OptionalGender = Annotated[Gender | None, _Blank]
OptionalAge = Annotated[Age | None, _Blank]
OptionalPitch = Annotated[Pitch | None, _Blank]
OptionalLanguage = Annotated[MainLanguage | None, _Blank]


def _idle_unload_seconds() -> float:
    raw = os.environ.get("OMNIVOICE_IDLE_UNLOAD_SECONDS", "60").strip()
    try:
        value = float(raw)
    except ValueError:
        return 60
    if value < 0:
        return 0
    return value


def load_settings() -> Settings:
    no_asr = os.environ.get("OMNIVOICE_NO_ASR", "").strip().lower()
    device = os.environ.get("OMNIVOICE_DEVICE", "").strip()
    return Settings(
        model=os.environ.get("OMNIVOICE_MODEL", "k2-fsa/OmniVoice"),
        device=device or None,
        no_asr=no_asr in {"1", "true", "yes", "y"},
        asr_model=os.environ.get(
            "OMNIVOICE_ASR_MODEL", "openai/whisper-large-v3-turbo"
        ),
        idle_unload_seconds=_idle_unload_seconds(),
    )


def _cuda_reserved_mib() -> float | None:
    if not torch.cuda.is_available() or not torch.cuda.is_initialized():
        return None
    return torch.cuda.memory_reserved() / (1024 * 1024)


def _to_cpu(module) -> None:
    if module is None:
        return
    try:
        from accelerate.hooks import remove_hook_from_submodules

        remove_hook_from_submodules(module)
    except Exception:
        pass
    try:
        module.to("cpu")
    except Exception:
        logger.debug("No se pudo mover un módulo a CPU", exc_info=True)


def _release_gpu(model: OmniVoice) -> None:
    try:
        pipe = getattr(model, "_asr_pipe", None)
        model._asr_pipe = None
        if pipe is not None:
            _to_cpu(getattr(pipe, "model", None))
            pipe.model = None
            del pipe
        tokenizer = getattr(model, "audio_tokenizer", None)
        model.audio_tokenizer = None
        _to_cpu(tokenizer)
        del tokenizer
        _to_cpu(model)
    finally:
        del model
        gc.collect()
        gc.collect()
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            ipc_collect = getattr(torch.cuda, "ipc_collect", None)
            if ipc_collect is not None:
                ipc_collect()


def _load_model() -> OmniVoice:
    settings: Settings = app.state.settings
    device = app.state.device
    logger.info("Cargando %s en %s", settings.model, device)
    model = OmniVoice.from_pretrained(
        settings.model,
        device_map=device,
        dtype=torch.float16,
        load_asr=not settings.no_asr,
        asr_model_name=settings.asr_model,
    )
    app.state.sampling_rate = int(model.sampling_rate)
    logger.info("Modelo listo en %s", device)
    return model


def _acquire_model() -> OmniVoice:
    global _model_users, _last_used
    with _model_lock:
        model = getattr(app.state, "model", None)
        if model is None:
            model = _load_model()
            app.state.model = model
        _model_users += 1
        _last_used = time.monotonic()
        return model


def _release_model_user() -> None:
    global _model_users, _last_used
    with _model_lock:
        _model_users = max(0, _model_users - 1)
        _last_used = time.monotonic()


def _unload_if_idle() -> None:
    settings: Settings | None = getattr(app.state, "settings", None)
    timeout = settings.idle_unload_seconds if settings is not None else 0
    if timeout <= 0:
        return
    with _model_lock:
        model = getattr(app.state, "model", None)
        if model is None or _model_users > 0:
            return
        if time.monotonic() - _last_used < timeout:
            return
        app.state.model = None
        before = _cuda_reserved_mib()
        _release_gpu(model)
        after = _cuda_reserved_mib()
    if before is None or after is None:
        logger.info("Sin actividad durante %.0f s. Modelo fuera de la VRAM.", timeout)
        return
    logger.info(
        "Sin actividad durante %.0f s. VRAM reservada: %.0f MiB -> %.0f MiB.",
        timeout,
        before,
        after,
    )


def _unload_now() -> None:
    with _model_lock:
        model = getattr(app.state, "model", None)
        app.state.model = None
        if model is None:
            return
        _release_gpu(model)


async def _sampling_rate() -> int:
    cached = getattr(app.state, "sampling_rate", None)
    if cached:
        return int(cached)
    model = await asyncio.to_thread(_acquire_model)
    try:
        return int(model.sampling_rate)
    finally:
        _release_model_user()


def _clean_language(language: str | None) -> str | None:
    if language is None:
        return None
    value = language.strip()
    if not value or value.lower() == "auto":
        return None
    return value


def build_instruct(
    *,
    gender: str | None,
    age: str | None,
    pitch: str | None,
) -> str | None:
    selected = [part for part in (gender, age, pitch) if part]
    if not selected:
        return None
    return ", ".join(selected)


def generation_fields(
    speed: float = Form(1.0, ge=0.5, le=1.5, description="1.0 es el ritmo normal."),
    duration: OptionalFloat = Form(
        None,
        gt=0,
        description="Duración fija en segundos. Si se define, reemplaza a speed.",
    ),
    num_step: int = Form(32, ge=4, le=64, description="Pasos de inferencia."),
    guidance_scale: float = Form(2.0, ge=0.0, le=4.0, description="Escala de guidance."),
    denoise: bool = Form(True, description="Activa el token de reducción de ruido."),
    preprocess_prompt: bool = Form(
        True,
        description="Recorta silencios del audio de referencia y ajusta la puntuación del texto de referencia.",
    ),
    postprocess_output: bool = Form(
        True,
        description="Quita silencios largos del audio generado.",
    ),
    normalize_text: bool = Form(
        False,
        description="Convierte números, fechas y montos a su forma hablada.",
    ),
) -> GenerationOptions:
    return GenerationOptions(
        speed=speed,
        duration=duration,
        num_step=num_step,
        guidance_scale=guidance_scale,
        denoise=denoise,
        preprocess_prompt=preprocess_prompt,
        postprocess_output=postprocess_output,
        normalize_text=normalize_text,
    )


def delivery_fields(
    generation_mode: GenerationMode = Form(
        "sync",
        description="sync espera el audio en esta petición. sse envía el progreso por Server-Sent Events. job responde al momento; el progreso se consulta en el estado del trabajo.",
    ),
    audio_response: AudioResponseMode = Form(
        "archivo",
        description="base64 devuelve el audio codificado. archivo lo descarga. url devuelve un enlace. En sse y job, archivo también se entrega como enlace de descarga.",
    ),
    audio_format: AudioFormat = Form(
        "wav",
        description="Formato del audio: wav, flac, ogg, mp3 o aiff.",
    ),
) -> DeliveryOptions:
    return DeliveryOptions(
        generation_mode=generation_mode,
        audio_response=audio_response,
        audio_format=audio_format,
    )


def _generation_config(options: GenerationOptions) -> OmniVoiceGenerationConfig:
    return OmniVoiceGenerationConfig(
        num_step=options.num_step,
        guidance_scale=options.guidance_scale,
        denoise=options.denoise,
        preprocess_prompt=options.preprocess_prompt,
        postprocess_output=options.postprocess_output,
    )


def _synthesis_kwargs(
    *,
    text: str,
    language: str | None,
    instruct: str | None,
    options: GenerationOptions,
    ref_text: str | None = None,
    ref_audio=None,
    voice_clone_prompt: VoiceClonePrompt | None = None,
) -> dict:
    cleaned = text.strip()
    if not cleaned:
        raise HTTPException(status_code=400, detail="El texto a sintetizar está vacío.")

    kwargs: dict = {
        "text": cleaned,
        "language": _clean_language(language),
        "generation_config": _generation_config(options),
        "normalize_text": options.normalize_text,
    }
    if instruct and instruct.strip():
        kwargs["instruct"] = instruct.strip()
    if options.speed != 1.0:
        kwargs["speed"] = options.speed
    if options.duration is not None and options.duration > 0:
        kwargs["duration"] = options.duration
    if ref_text and ref_text.strip():
        kwargs["ref_text"] = ref_text.strip()
    if ref_audio is not None:
        kwargs["ref_audio"] = ref_audio
    if voice_clone_prompt is not None:
        kwargs["voice_clone_prompt"] = voice_clone_prompt
    return kwargs


def _progress_payload(step: int, total: int) -> dict:
    if total <= 0:
        percent = 0
    else:
        percent = round(min(step, total) / total * 100)
    return {"step": step, "total_steps": total, "progress": percent}


def _encode_audio(
    kwargs: dict,
    audio_format: AudioFormat,
    on_progress=None,
) -> tuple[bytes, int]:
    model = _acquire_model()
    try:
        try:
            with _infer_lock:
                audios = model.generate(**kwargs, on_progress=on_progress)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Error al generar audio")
            raise HTTPException(
                status_code=500,
                detail=f"{type(exc).__name__}: {exc}",
            ) from exc

        if not audios:
            raise HTTPException(status_code=500, detail="El modelo no devolvió audio.")

        spec = _AUDIO_FORMATS[audio_format]
        buffer = io.BytesIO()
        sampling_rate = int(model.sampling_rate)
        try:
            sf.write(
                buffer,
                np.asarray(audios[0]),
                sampling_rate,
                format=spec["soundfile"],
            )
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"No se pudo escribir el audio en {audio_format}: {exc}",
            ) from exc
        return buffer.getvalue(), sampling_rate
    finally:
        _release_model_user()


def _audio_response(payload: bytes, sampling_rate: int, filename: str, media_type: str) -> Response:
    return Response(
        content=payload,
        media_type=media_type,
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "X-Sampling-Rate": str(sampling_rate),
        },
    )


def _base_url(request: Request) -> str:
    root = request.scope.get("root_path") or ""
    return str(request.base_url).rstrip("/") + root


def _save_audio(payload: bytes, extension: str) -> str:
    _AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    audio_id = uuid.uuid4().hex
    name = f"{audio_id}.{extension}"
    (_AUDIO_DIR / name).write_bytes(payload)
    return name


def _pack_audio(
    payload: bytes,
    sampling_rate: int,
    audio_response: AudioResponseMode,
    audio_format: AudioFormat,
    base_url: str,
) -> dict:
    spec = _AUDIO_FORMATS[audio_format]
    body: dict = {
        "sampling_rate": sampling_rate,
        "media_type": spec["media_type"],
        "audio_format": audio_format,
        "audio_response": audio_response,
    }
    if audio_response == "base64":
        body["audio_base64"] = base64.b64encode(payload).decode("ascii")
        return body
    name = _save_audio(payload, spec["extension"])
    body["audio_id"] = name
    body["url"] = f"{base_url}/v1/audio/{name}"
    return body


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _error_detail(exc: HTTPException) -> str:
    detail = exc.detail
    if isinstance(detail, str):
        return detail
    return json.dumps(detail, ensure_ascii=False)


async def _sse_events(
    kwargs: dict,
    audio_response: AudioResponseMode,
    audio_format: AudioFormat,
    base_url: str,
):
    yield _sse("status", {"state": "started"})
    yield _sse("status", {"state": "generating"})
    yield _sse("progress", _progress_payload(0, 0))
    progress_queue: queue.Queue[dict] = queue.Queue()

    def on_progress(step: int, total: int) -> None:
        progress_queue.put(_progress_payload(step, total))

    task = asyncio.create_task(
        asyncio.to_thread(_encode_audio, kwargs, audio_format, on_progress)
    )
    loop = asyncio.get_running_loop()
    last_keepalive = loop.time()
    while not task.done() or not progress_queue.empty():
        emitted = False
        while True:
            try:
                item = progress_queue.get_nowait()
            except queue.Empty:
                break
            yield _sse("progress", item)
            emitted = True
        if task.done() and progress_queue.empty():
            break
        if emitted:
            last_keepalive = loop.time()
            continue
        await asyncio.wait({task}, timeout=0.25)
        now = loop.time()
        if not task.done() and now - last_keepalive >= 5:
            yield ": keepalive\n\n"
            last_keepalive = now
    try:
        payload, sampling_rate = task.result()
    except HTTPException as exc:
        yield _sse("error", {"detail": _error_detail(exc)})
        return
    except Exception as exc:
        logger.exception("Error al generar audio por SSE")
        yield _sse("error", {"detail": f"{type(exc).__name__}: {exc}"})
        return
    yield _sse("result", _pack_audio(payload, sampling_rate, audio_response, audio_format, base_url))
    yield _sse("done", {"state": "completed"})


def _run_job(
    job_id: str,
    kwargs: dict,
    audio_response: AudioResponseMode,
    audio_format: AudioFormat,
    base_url: str,
) -> None:
    with _jobs_lock:
        _jobs[job_id]["status"] = "running"
        _jobs[job_id].update(_progress_payload(0, 0))

    def on_progress(step: int, total: int) -> None:
        with _jobs_lock:
            job = _jobs.get(job_id)
            if job is not None:
                job.update(_progress_payload(step, total))

    try:
        payload, sampling_rate = _encode_audio(kwargs, audio_format, on_progress)
        result = _pack_audio(payload, sampling_rate, audio_response, audio_format, base_url)
    except HTTPException as exc:
        with _jobs_lock:
            _jobs[job_id]["status"] = "failed"
            _jobs[job_id]["error"] = _error_detail(exc)
        return
    except Exception as exc:
        logger.exception("Error en el trabajo %s", job_id)
        with _jobs_lock:
            _jobs[job_id]["status"] = "failed"
            _jobs[job_id]["error"] = f"{type(exc).__name__}: {exc}"
        return
    with _jobs_lock:
        job = _jobs[job_id]
        job["status"] = "completed"
        job["sampling_rate"] = sampling_rate
        job["result"] = result
        job["progress"] = 100
        if job["total_steps"]:
            job["step"] = job["total_steps"]


def _start_job(
    kwargs: dict,
    audio_response: AudioResponseMode,
    audio_format: AudioFormat,
    base_url: str,
) -> JSONResponse:
    job_id = uuid.uuid4().hex
    with _jobs_lock:
        _jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "error": None,
            "sampling_rate": None,
            "audio_response": audio_response,
            "audio_format": audio_format,
            "progress": 0,
            "step": 0,
            "total_steps": 0,
            "result": None,
        }
    threading.Thread(
        target=_run_job,
        args=(job_id, kwargs, audio_response, audio_format, base_url),
        daemon=True,
    ).start()
    return JSONResponse(
        status_code=202,
        content={
            "job_id": job_id,
            "status": "queued",
            "status_url": f"{base_url}/v1/jobs/{job_id}",
        },
    )


async def _deliver(
    request: Request,
    kwargs: dict,
    filename: str,
    delivery: DeliveryOptions,
):
    base_url = _base_url(request)
    spec = _AUDIO_FORMATS[delivery.audio_format]
    filename = f"{filename}.{spec['extension']}"
    if delivery.generation_mode == "job":
        return _start_job(kwargs, delivery.audio_response, delivery.audio_format, base_url)
    if delivery.generation_mode == "sse":
        return StreamingResponse(
            _sse_events(kwargs, delivery.audio_response, delivery.audio_format, base_url),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    payload, sampling_rate = await asyncio.to_thread(
        _encode_audio, kwargs, delivery.audio_format
    )
    if delivery.audio_response == "archivo":
        return _audio_response(payload, sampling_rate, filename, spec["media_type"])
    return _pack_audio(
        payload, sampling_rate, delivery.audio_response, delivery.audio_format, base_url
    )


def _voice_labels() -> list[tuple[str, str]]:
    voices: list[SavedVoice] = []
    if _VOICES_DIR.is_dir():
        for folder in _VOICES_DIR.iterdir():
            if not folder.is_dir():
                continue
            voice = _read_voice(folder)
            if voice is not None:
                voices.append(voice)
    voices.sort(key=lambda voice: voice.name.casefold())
    used: set[str] = set()
    labels: list[tuple[str, str]] = []
    for voice in voices:
        label = voice.name.replace("/", " ").replace("?", " ").replace("#", " ").strip()
        if not label:
            label = "voz"
        if label in used:
            label = f"{label} ({voice.id[:8]})"
        suffix = 2
        base = label
        while label in used:
            label = f"{base}-{suffix}"
            suffix += 1
        used.add(label)
        labels.append((label, voice.id))
    return labels


def _resolve_voice_id(value: str) -> str:
    if _VOICE_ID_RE.fullmatch(value):
        return value
    for label, voice_id in _voice_labels():
        if label == value:
            return voice_id
    raise HTTPException(status_code=404, detail="Voz no encontrada.")


def _voice_dir(voice_id: str) -> Path:
    if not _VOICE_ID_RE.fullmatch(voice_id):
        raise HTTPException(status_code=404, detail="Voz no encontrada.")
    return _VOICES_DIR / voice_id


def _read_voice(folder: Path) -> SavedVoice | None:
    meta_path = folder / "meta.json"
    prompt_path = folder / "prompt.pt"
    if not meta_path.is_file() or not prompt_path.is_file():
        return None
    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        return SavedVoice.model_validate(data)
    except (OSError, json.JSONDecodeError, ValueError):
        return None


def _create_saved_voice(
    waveform: np.ndarray,
    name: str,
    reference_text: str | None,
    language: MainLanguage,
) -> SavedVoice:
    model = _acquire_model()
    try:
        try:
            with _infer_lock:
                prompt = model.create_voice_clone_prompt(
                    ref_audio=(torch.from_numpy(waveform), model.sampling_rate),
                    ref_text=reference_text,
                    preprocess_prompt=True,
                )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Error al guardar la voz")
            raise HTTPException(
                status_code=500,
                detail=f"{type(exc).__name__}: {exc}",
            ) from exc

        voice_id = uuid.uuid4().hex
        folder = _VOICES_DIR / voice_id
        folder.mkdir(parents=True, exist_ok=False)
        try:
            prompt.save(str(folder / "prompt.pt"))
            voice = SavedVoice(
                id=voice_id,
                name=name,
                created_at=datetime.now(timezone.utc).isoformat(),
                reference_text=prompt.ref_text or "",
                language=language,
            )
            (folder / "meta.json").write_text(
                voice.model_dump_json(),
                encoding="utf-8",
            )
        except Exception:
            shutil.rmtree(folder, ignore_errors=True)
            raise
        return voice
    finally:
        _release_model_user()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _last_used
    settings = load_settings()
    device = settings.device or get_best_device()
    app.state.model = None
    app.state.device = device
    app.state.settings = settings
    app.state.sampling_rate = None
    _last_used = time.monotonic()
    if settings.idle_unload_seconds <= 0:
        app.state.model = await asyncio.to_thread(_load_model)
        logger.info("Modelo residente en %s. Swagger en /docs", device)
    else:
        logger.info(
            "API lista. %s entra en %s con la primera petición y sale de la VRAM tras %.0f s sin uso. Swagger en /docs",
            settings.model,
            device,
            settings.idle_unload_seconds,
        )
    stop = asyncio.Event()

    async def _watch_idle() -> None:
        while True:
            try:
                await asyncio.wait_for(stop.wait(), timeout=_UNLOAD_POLL_SECONDS)
                return
            except asyncio.TimeoutError:
                pass
            try:
                await asyncio.to_thread(_unload_if_idle)
            except Exception:
                logger.exception("No se pudo liberar la VRAM")

    watcher = asyncio.create_task(_watch_idle())
    try:
        yield
    finally:
        stop.set()
        await watcher
        for _ in range(20):
            with _model_lock:
                busy = _model_users > 0
            if not busy:
                break
            await asyncio.sleep(0.25)
        await asyncio.to_thread(_unload_now)


app = FastAPI(
    title="API Voice Clone",
    version=__version__,
    description=(
        "Síntesis de voz con OmniVoice: clonación a partir de un audio de referencia, "
        "voces guardadas y síntesis con género, edad y tono. "
        "Cada síntesis puede esperar el resultado, emitirlo por SSE o dejarlo en un trabajo, "
        "y devolver el audio como archivo, base64 o URL en wav, flac, ogg, mp3 o aiff. "
        "Sin peticiones, OmniVoice y Whisper salen de la VRAM y se vuelven a cargar en la siguiente síntesis."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)


def _inject_voice_select(schema: dict) -> None:
    labels = [label for label, _voice_id in _voice_labels()]
    for path_item in schema.get("paths", {}).values():
        for method, operation in path_item.items():
            if method not in {"get", "post", "put", "patch", "delete"}:
                continue
            for param in operation.get("parameters") or []:
                if param.get("name") != "voice_id":
                    continue
                param_schema = param.setdefault("schema", {})
                if labels:
                    param_schema["type"] = "string"
                    param_schema["enum"] = labels
                    param["description"] = (
                        "Elige una voz guardada. El selector se actualiza cada vez que se abre esta página."
                    )
                else:
                    param_schema.pop("enum", None)
                    param["description"] = (
                        "No hay voces guardadas. Crea una con POST /v1/voices y vuelve a abrir esta página."
                    )


def _openapi() -> dict:
    schema = get_openapi(
        title=app.title,
        version=app.version,
        openapi_version=app.openapi_version,
        description=app.description,
        routes=app.routes,
    )
    _inject_voice_select(schema)
    return schema


app.openapi = _openapi


@app.middleware("http")
async def _docs_without_cache(request: Request, call_next):
    response = await call_next(request)
    if request.url.path in {app.openapi_url, app.docs_url, app.redoc_url}:
        response.headers["Cache-Control"] = "no-store"
    return response

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


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse, tags=["Sistema"], summary="Estado del servicio")
def health():
    settings: Settings = getattr(app.state, "settings", None) or load_settings()
    model = getattr(app.state, "model", None)
    return HealthResponse(
        status="ok",
        model=settings.model,
        device=str(getattr(app.state, "device", "") or ""),
        sampling_rate=getattr(app.state, "sampling_rate", None),
        asr_enabled=not settings.no_asr,
        model_loaded=model is not None,
    )


@app.get(
    "/v1/languages",
    response_model=LanguageList,
    tags=["Catálogo"],
    summary="Idiomas admitidos",
)
def languages():
    return LanguageList(languages=sorted(lang_display_name(name) for name in LANG_NAMES))


@app.get(
    "/v1/voices",
    response_model=SavedVoiceList,
    tags=["Voces"],
    summary="Listar voces guardadas",
)
def list_voices():
    voices: list[SavedVoice] = []
    if _VOICES_DIR.is_dir():
        for folder in _VOICES_DIR.iterdir():
            if folder.is_dir():
                voice = _read_voice(folder)
                if voice is not None:
                    voices.append(voice)
    voices.sort(key=lambda voice: voice.created_at, reverse=True)
    return SavedVoiceList(voices=voices)


@app.post(
    "/v1/voices",
    response_model=SavedVoice,
    status_code=201,
    tags=["Voces"],
    summary="Guardar una voz clonada",
)
async def create_voice(
    name: str = Form(..., description="Nombre para reconocer la voz en el listado."),
    reference_audio: UploadFile = File(
        ...,
        description="Audio de referencia. Lo recomendable son 3 a 10 segundos.",
    ),
    language: MainLanguage = Form(
        ...,
        description="Idioma de la voz: English, Chinese, Japanese, Spanish o French.",
    ),
    reference_text: OptionalStr = Form(
        None,
        description="Transcripción del audio. Si se omite, Whisper la transcribe.",
    ),
):
    """Codifica el audio una vez y guarda la voz para reutilizarla."""
    cleaned_name = name.strip()
    if not cleaned_name:
        raise HTTPException(status_code=400, detail="El nombre de la voz está vacío.")
    if len(cleaned_name) > 80:
        raise HTTPException(status_code=400, detail="El nombre admite hasta 80 caracteres.")
    settings: Settings = getattr(app.state, "settings", None) or load_settings()
    transcript = reference_text.strip() if reference_text and reference_text.strip() else None
    if transcript is None and settings.no_asr:
        raise HTTPException(
            status_code=400,
            detail="Hace falta la transcripción porque Whisper no está cargado.",
        )
    raw = await reference_audio.read()
    if not raw:
        raise HTTPException(status_code=400, detail="El audio de referencia está vacío.")
    if len(raw) > MAX_AUDIO_BYTES:
        raise HTTPException(
            status_code=413,
            detail="El audio de referencia supera los 25 MB.",
        )
    sampling_rate = await _sampling_rate()
    try:
        waveform = load_audio_bytes(raw, sampling_rate)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"No se pudo leer el audio de referencia: {exc}",
        ) from exc
    return await asyncio.to_thread(
        _create_saved_voice, waveform, cleaned_name, transcript, language
    )


@app.delete(
    "/v1/voices/{voice_id}",
    status_code=204,
    tags=["Voces"],
    summary="Eliminar una voz guardada",
)
def delete_voice(voice_id: str):
    voice_id = _resolve_voice_id(voice_id)
    folder = _voice_dir(voice_id)
    if not folder.is_dir() or _read_voice(folder) is None:
        raise HTTPException(status_code=404, detail="Voz no encontrada.")
    shutil.rmtree(folder)
    return Response(status_code=204)


def _load_saved_prompt(voice_id: str) -> VoiceClonePrompt:
    folder = _voice_dir(voice_id)
    if not folder.is_dir() or _read_voice(folder) is None:
        raise HTTPException(status_code=404, detail="Voz no encontrada.")
    try:
        return VoiceClonePrompt.load(str(folder / "prompt.pt"))
    except Exception as exc:
        logger.exception("Error al cargar la voz %s", voice_id)
        raise HTTPException(
            status_code=500,
            detail=f"No se pudo cargar la voz guardada: {exc}",
        ) from exc


@app.post(
    "/v1/voices/{voice_id}/synthesize",
    tags=["Voces"],
    summary="Sintetizar con una voz guardada",
    responses=_SYNTHESIS_RESPONSES,
)
async def synthesize_saved_voice(
    voice_id: str,
    request: Request,
    text: str = Form(..., description="Texto a sintetizar."),
    options: GenerationOptions = Depends(generation_fields),
    delivery: DeliveryOptions = Depends(delivery_fields),
):
    """Usa una voz ya guardada y el idioma elegido al crearla."""
    voice_id = _resolve_voice_id(voice_id)
    voice = _read_voice(_voice_dir(voice_id))
    if voice is None:
        raise HTTPException(status_code=404, detail="Voz no encontrada.")
    prompt = _load_saved_prompt(voice_id)
    kwargs = _synthesis_kwargs(
        text=text,
        language=voice.language,
        instruct=None,
        options=options,
        voice_clone_prompt=prompt,
    )
    return await _deliver(request, kwargs, "voice", delivery)


@app.get("/v1/audio/{audio_name}", tags=["Síntesis"], summary="Descargar un audio generado")
def download_audio(audio_name: str):
    if not _AUDIO_NAME_RE.fullmatch(audio_name):
        raise HTTPException(status_code=404, detail="Audio no encontrado.")
    path = _AUDIO_DIR / audio_name
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Audio no encontrado.")
    media_type = _AUDIO_FORMATS[path.suffix[1:]]["media_type"]
    return FileResponse(path, media_type=media_type, filename=audio_name)


@app.get("/v1/jobs/{job_id}", tags=["Síntesis"], summary="Estado de un trabajo")
def job_status(job_id: str):
    with _jobs_lock:
        record = _jobs.get(job_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Trabajo no encontrado.")
        result = dict(record["result"]) if record["result"] else None
        body = {
            "job_id": record["job_id"],
            "status": record["status"],
            "error": record["error"],
            "sampling_rate": record["sampling_rate"],
            "audio_response": record["audio_response"],
            "audio_format": record["audio_format"],
            "progress": record["progress"],
            "step": record["step"],
            "total_steps": record["total_steps"],
        }
    if result:
        for key in ("media_type", "audio_base64", "url", "audio_id"):
            if key in result:
                body[key] = result[key]
    return body


@app.post(
    "/v1/synthesize",
    tags=["Síntesis"],
    summary="Sintetizar",
    responses=_SYNTHESIS_RESPONSES,
)
async def synthesize(
    request: Request,
    text: str = Form(..., description="Texto a sintetizar."),
    language: OptionalLanguage = Form(
        None,
        description="English, Chinese, Japanese, Spanish o French. Vacío detecta el idioma.",
    ),
    gender: OptionalGender = Form(
        None,
        description="Género de la voz: male o female. Vacío deja que el modelo elija la voz.",
    ),
    age: OptionalAge = Form(
        None,
        description="Edad de la voz: child, teenager, young adult, middle-aged o elderly.",
    ),
    pitch: OptionalPitch = Form(
        None,
        description="Tono: very low pitch, low pitch, moderate pitch, high pitch o very high pitch.",
    ),
    options: GenerationOptions = Depends(generation_fields),
    delivery: DeliveryOptions = Depends(delivery_fields),
):
    """Sintetiza el texto. Género, edad y tono afinan la voz; vacíos, el modelo elige."""
    kwargs = _synthesis_kwargs(
        text=text,
        language=language,
        instruct=build_instruct(gender=gender, age=age, pitch=pitch),
        options=options,
    )
    return await _deliver(request, kwargs, "speech", delivery)


@app.post(
    "/v1/voice-clone",
    tags=["Síntesis"],
    summary="Clonar una voz",
    responses=_SYNTHESIS_RESPONSES,
)
async def voice_clone(
    request: Request,
    text: str = Form(..., description="Texto a sintetizar."),
    reference_audio: UploadFile = File(
        ...,
        description="Audio de referencia. Lo recomendable son 3 a 10 segundos.",
    ),
    reference_text: OptionalStr = Form(
        None,
        description="Transcripción del audio de referencia. Si se omite, Whisper la transcribe.",
    ),
    language: OptionalLanguage = Form(
        None,
        description="English, Chinese, Japanese, Spanish o French. Vacío detecta el idioma.",
    ),
    options: GenerationOptions = Depends(generation_fields),
    delivery: DeliveryOptions = Depends(delivery_fields),
):
    """Clona la voz de un audio de referencia y sintetiza el texto."""
    raw = await reference_audio.read()
    if not raw:
        raise HTTPException(status_code=400, detail="El audio de referencia está vacío.")
    if len(raw) > MAX_AUDIO_BYTES:
        raise HTTPException(
            status_code=413,
            detail="El audio de referencia supera los 25 MB.",
        )
    sampling_rate = await _sampling_rate()
    try:
        waveform = load_audio_bytes(raw, sampling_rate)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"No se pudo leer el audio de referencia: {exc}",
        ) from exc

    kwargs = _synthesis_kwargs(
        text=text,
        language=language,
        instruct=None,
        options=options,
        ref_text=reference_text,
        ref_audio=(torch.from_numpy(waveform), sampling_rate),
    )
    return await _deliver(request, kwargs, "voice-clone", delivery)
