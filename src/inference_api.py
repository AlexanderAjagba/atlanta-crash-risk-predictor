"""Small HTTP adapter for model-based Atlanta area ranking."""
from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from src.models.area_ranking import rank_area_candidates
from src.models.hf_inference import load_model


ROOT = Path(__file__).resolve().parents[1]
GRID_PATH = ROOT / "frontend" / "src" / "research-data.json"


def load_grid_candidates(path: Path = GRID_PATH) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [
        {
            "id": cell["id"],
            "latitude": cell["latitude"],
            "longitude": cell["longitude"],
        }
        for cell in data["cells"]
    ]


class InferenceHandler(BaseHTTPRequestHandler):
    model: Any = None
    candidates: list[dict[str, Any]] = []

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._send_json(200, {"status": "ok", "candidateCount": len(self.candidates)})
            return
        self._send_json(404, {"error": "Not found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/areas/rank":
            self._send_json(404, {"error": "Not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            request = json.loads(self.rfile.read(length))
            scenario_datetime = request["scenarioDatetime"]
            top_k = request.get("topK")
            ranked = rank_area_candidates(
                self.model,
                self.candidates,
                scenario_datetime,
                top_k,
            )
            self._send_json(
                200,
                {
                    "scenarioDatetime": scenario_datetime,
                    "conditionalOnRecordedCrash": True,
                    "areas": ranked,
                },
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            self._send_json(400, {"error": str(error)})
        except Exception as error:  # Keep model/service failures explicit to callers.
            self._send_json(500, {"error": f"Inference failed: {error}"})

    def log_message(self, format: str, *args: Any) -> None:
        return


def create_server(model: Any, candidates: list[dict[str, Any]], host: str, port: int) -> ThreadingHTTPServer:
    handler = type("ConfiguredInferenceHandler", (InferenceHandler,), {})
    handler.model = model
    handler.candidates = candidates
    return ThreadingHTTPServer((host, port), handler)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--local-model")
    parser.add_argument("--repo-id")
    parser.add_argument("--filename", default="final_severity_logistic_2021_2024.joblib")
    parser.add_argument("--revision")
    args = parser.parse_args()
    model = load_model(args.repo_id, args.filename, args.revision, args.local_model)
    server = create_server(model, load_grid_candidates(), args.host, args.port)
    print(f"Inference service listening on http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()