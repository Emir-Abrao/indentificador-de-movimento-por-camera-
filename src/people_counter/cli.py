"""CLI with explicit camera access and opt-in model/video downloads or recording."""

import argparse
import importlib.metadata
import json
import logging
import os
import sqlite3
import sys

from . import __version__
from .config import load_settings
from .storage import export_csv


def main(argv=None):
    parser = argparse.ArgumentParser(description="Contador local de pessoas por camera")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    execute = commands.add_parser("run", help="Iniciar camera, video ou RTSP")
    execute.add_argument("--config")
    execute.add_argument("--source", default="0", help="Indice da webcam ou caminho do video")
    execute.add_argument("--source-env", help="Variavel com URL RTSP (evita senha no historico)")
    execute.add_argument("--detector", choices=["yolo", "hog"])
    execute.add_argument("--model")
    execute.add_argument("--device")
    execute.add_argument("--confidence", type=float)
    execute.add_argument("--headless", action="store_true")
    execute.add_argument("--max-frames", type=int)
    execute.add_argument("--database", default="data/counts.db")
    execute.add_argument("--output-video", help="Opt-in: gravar MP4 anotado (nao sobrescreve)")
    execute.add_argument("--allow-model-download", action="store_true")
    export = commands.add_parser("export", help="Exportar eventos SQLite para CSV")
    export.add_argument("--database", default="data/counts.db")
    export.add_argument("--output", required=True)
    commands.add_parser("doctor", help="Diagnostico local sem abrir camera ou baixar modelo")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        if args.command == "doctor":
            result = {"python": sys.version.split()[0], "app": __version__}
            for name in (
                "numpy",
                "scipy",
                "opencv-python",
                "opencv-python-headless",
                "ultralytics",
            ):
                try:
                    result[name] = importlib.metadata.version(name)
                except importlib.metadata.PackageNotFoundError:
                    result[name] = "not installed"
            print(json.dumps(result, indent=2))
            return 0
        if args.command == "export":
            count = export_csv(args.database, args.output)
            print(json.dumps({"events_exported": count, "output": args.output}))
            return 0
        settings = load_settings(
            args.config,
            detector=args.detector,
            model=args.model,
            device=args.device,
            confidence=args.confidence,
        )
        source = os.environ.get(args.source_env, "") if args.source_env else args.source
        if not source:
            raise ValueError("Camera source environment variable is empty")
        from .app import run

        summary = run(
            settings,
            source,
            args.database,
            args.headless,
            args.max_frames,
            args.output_video,
            args.allow_model_download,
        )
        print(json.dumps(summary, indent=2))
        return 0
    except (ValueError, OSError, RuntimeError, sqlite3.Error, ImportError) as exc:
        # Sources are never intentionally interpolated. Suppress provider errors that
        # could contain URL credentials; actionable validation messages stay local.
        message = str(exc)
        if "rtsp" in message.lower() or "://" in message:
            message = "Source/provider failure; credentials omitted. Check configuration."
        logging.error("%s", message)
        return 2
