"""Arranque de la API: ``python -m omnivoice.api``."""

from __future__ import annotations

import argparse
import logging
import os

import uvicorn


def main(argv=None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    parser = argparse.ArgumentParser(
        prog="omnivoice-api",
        description="API HTTP de OmniVoice con Swagger en /docs.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("OMNIVOICE_MODEL", "k2-fsa/OmniVoice"),
        help="Ruta del checkpoint o id de Hugging Face.",
    )
    parser.add_argument(
        "--device",
        default=os.environ.get("OMNIVOICE_DEVICE") or None,
        help="Dispositivo. Si se omite, se detecta solo.",
    )
    parser.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("PORT", "3036")),
    )
    parser.add_argument(
        "--no-asr",
        action="store_true",
        default=os.environ.get("OMNIVOICE_NO_ASR", "").lower() in {"1", "true", "yes"},
        help="No carga Whisper. Sin esto no se puede transcribir la referencia.",
    )
    parser.add_argument(
        "--asr-model",
        default=os.environ.get("OMNIVOICE_ASR_MODEL", "openai/whisper-large-v3-turbo"),
    )
    parser.add_argument(
        "--idle-unload-seconds",
        type=float,
        default=float(os.environ.get("OMNIVOICE_IDLE_UNLOAD_SECONDS", "60")),
        help="Segundos sin peticiones antes de soltar la VRAM. 0 mantiene el modelo cargado.",
    )
    parser.add_argument(
        "--root-path",
        default=os.environ.get("ROOT_PATH") or None,
        help="Prefijo si la API queda detrás de un proxy inverso.",
    )
    args = parser.parse_args(argv)

    os.environ["OMNIVOICE_MODEL"] = args.model
    if args.device:
        os.environ["OMNIVOICE_DEVICE"] = args.device
    os.environ["OMNIVOICE_NO_ASR"] = "1" if args.no_asr else "0"
    os.environ["OMNIVOICE_ASR_MODEL"] = args.asr_model
    os.environ["OMNIVOICE_IDLE_UNLOAD_SECONDS"] = str(args.idle_unload_seconds)

    uvicorn.run(
        "omnivoice.api.app:app",
        host=args.host,
        port=args.port,
        root_path=args.root_path or "",
        log_level="info",
    )


if __name__ == "__main__":
    main()
