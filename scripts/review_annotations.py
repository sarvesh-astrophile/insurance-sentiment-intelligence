"""Local-only annotation UI. Run: .venv/bin/python scripts/review_annotations.py"""
import argparse
import csv
import hashlib
import io
import json
import secrets
import sys
import tempfile
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from insurance_sentiment_analysis import clean_text

LABELS = {"positive", "negative", "neutral", "mixed_demo"}


class Store:
    def __init__(self, root=ROOT):
        self.root = Path(root)
        self.path = self.root / "data/daa_annotations.csv"
        self.lock = threading.Lock()

    def read(self):
        content = self.path.read_bytes()
        rows = list(csv.DictReader(io.StringIO(content.decode("utf-8"))))
        return rows, hashlib.sha256(content).hexdigest()

    def snapshot(self):
        rows, revision = self.read()
        with (self.root / "data/daa_reviews_sample.csv").open(encoding="utf-8", newline="") as handle:
            texts = {row["review_id"]: clean_text(row["text"]) for row in csv.DictReader(handle)}
        for row in rows:
            if row["review_id"] not in texts:
                raise ValueError("An annotation ID is missing from the sample")
            row["text"] = texts[row["review_id"]]
            if hashlib.sha256(row["text"].encode()).hexdigest() != row["text_sha256"]:
                raise ValueError("Review text changed; reconcile frozen hashes before annotation")
        return {"reviews": rows, "revision": revision}

    def save(self, payload):
        with self.lock:
            snapshot = self.snapshot()
            if payload.get("revision") != snapshot["revision"]:
                raise FileExistsError("Annotations changed elsewhere. Reload before saving; your form is unchanged.")
            reviewer = payload.get("reviewer", "")
            label = payload.get("human_sentiment", "")
            approved = payload.get("human_approved")
            if not isinstance(reviewer, str) or not reviewer.strip() or len(reviewer) > 100:
                raise ValueError("Enter reviewer initials/name (1–100 characters)")
            if label not in LABELS or not isinstance(approved, bool):
                raise ValueError("Select a valid sentiment and an explicit approval state")
            rows, _ = self.read()
            matches = [row for row in rows if row["review_id"] == payload.get("review_id")]
            if len(matches) != 1:
                raise ValueError("Unknown or duplicate review ID")
            # Preserve AI drafts, original notes, frozen text hashes and CSV schema.
            row = matches[0]
            row.update(human_sentiment=label, human_approved=str(approved), reviewer=reviewer.strip())
            backup = self.root / ".cache/annotation_backups"
            backup.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            (backup / f"{stamp}_{snapshot['revision'][:12]}.csv").write_bytes(self.path.read_bytes())
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=self.path.parent,
                                                 prefix=".annotations-", suffix=".tmp", delete=False) as handle:
                    temporary = Path(handle.name)
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
                temporary.replace(self.path)
            finally:
                if temporary and temporary.exists():
                    temporary.unlink()
            return self.snapshot()


def make_handler(store, token):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass  # Do not log annotation text or reviewer identity.

        def respond(self, status, value, content_type="application/json; charset=utf-8"):
            data = json.dumps(value).encode() if content_type.startswith("application/json") else value
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(data)

        def safe_host(self):
            return self.headers.get("Host") in {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}

        def do_GET(self):
            if not self.safe_host():
                return self.respond(403, {"error": "Localhost access only"})
            path = urlsplit(self.path).path
            assets = {"/": ("index.html", "text/html; charset=utf-8"),
                      "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                      "/style.css": ("style.css", "text/css; charset=utf-8")}
            if path in assets:
                name, mime = assets[path]
                return self.respond(200, (ROOT / "review_app" / name).read_bytes(), mime)
            if path == "/api/reviews":
                try:
                    return self.respond(200, {**store.snapshot(), "token": token})
                except ValueError as error:
                    return self.respond(400, {"error": str(error)})
            return self.respond(404, {"error": "Not found"})

        def do_POST(self):
            if not self.safe_host() or self.headers.get("X-Review-Token") != token:
                return self.respond(403, {"error": "Invalid local session; reload the page"})
            if self.path != "/api/reviews":
                return self.respond(404, {"error": "Not found"})
            if self.headers.get("Content-Type") != "application/json":
                return self.respond(415, {"error": "JSON required"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 10000:
                    raise ValueError("Invalid request size")
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError("JSON object required")
                return self.respond(200, store.save(payload))
            except FileExistsError as error:
                return self.respond(409, {"error": str(error)})
            except (ValueError, TypeError) as error:
                return self.respond(400, {"error": str(error)})
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    store = Store()
    store.snapshot()  # Validate selection and frozen input before starting.
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(store, secrets.token_urlsafe(32)))
    print(f"Review app: http://127.0.0.1:{server.server_port}", flush=True)
    print("Saves update data/daa_annotations.csv; backups: .cache/annotation_backups/", flush=True)
    print("No Jev calls. Press Ctrl+C to stop.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
