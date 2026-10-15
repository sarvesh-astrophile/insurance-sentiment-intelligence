"""Annotation persistence and local HTTP security tests; use temporary CSV copies."""
import json
import shutil
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from scripts.review_annotations import ROOT, Store, make_handler


class ReviewAppTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "data").mkdir()
        for name in ["daa_annotations.csv", "daa_reviews_sample.csv"]:
            shutil.copyfile(ROOT / "data" / name, self.root / "data" / name)
        self.store = Store(self.root)

    def payload(self, snapshot):
        return {"review_id": snapshot["reviews"][0]["review_id"], "revision": snapshot["revision"],
                "human_sentiment": "neutral", "human_approved": True, "reviewer": "TEST"}

    def test_save_preserves_drafts_and_creates_backup(self):
        before = self.store.snapshot()
        original = self.store.path.read_bytes()
        after = self.store.save(self.payload(before))
        first = after["reviews"][0]
        self.assertEqual(first["human_sentiment"], "neutral")
        self.assertEqual(first["human_approved"], "True")
        for field in ["draft_sentiment", "annotation_notes", "text_sha256"]:
            self.assertEqual(first[field], before["reviews"][0][field])
        self.assertEqual(after["reviews"][1:], before["reviews"][1:])
        backups = list((self.root / ".cache/annotation_backups").glob("*.csv"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_bytes(), original)
        # Standard pipeline can immediately read the updated CSV.
        from insurance_sentiment_analysis import attach_annotations, load_reviews
        reviews, _ = load_reviews(self.root / "data/daa_reviews_sample.csv")
        result = attach_annotations(reviews, self.store.path)
        self.assertEqual(result.gold_sentiment.iloc[0], "neutral")

    def test_stale_revision_does_not_overwrite(self):
        payload = self.payload(self.store.snapshot())
        self.store.save(payload)
        saved = self.store.path.read_bytes()
        with self.assertRaises(FileExistsError):
            self.store.save(payload)
        self.assertEqual(saved, self.store.path.read_bytes())

    def test_validation_does_not_write(self):
        before = self.store.path.read_bytes()
        for change in [{"reviewer": ""}, {"human_sentiment": "invalid"},
                       {"human_approved": "True"}, {"review_id": "missing"}]:
            payload = self.payload(self.store.snapshot()) | change
            with self.assertRaises(ValueError):
                self.store.save(payload)
        self.assertEqual(before, self.store.path.read_bytes())

    def test_unapproved_save_is_not_gold(self):
        payload = self.payload(self.store.snapshot()) | {"human_approved": False}
        self.assertEqual(self.store.save(payload)["reviews"][0]["human_approved"], "False")

    def test_http_get_save_and_csrf(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(self.store, "test-token"))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            url = f"http://127.0.0.1:{server.server_port}"
            with urlopen(url + "/api/reviews") as response:
                data = json.load(response)
            self.assertEqual(len(data["reviews"]), 50)
            body = json.dumps(self.payload(data)).encode()
            bad = Request(url + "/api/reviews", data=body, headers={"Content-Type": "application/json"})
            with self.assertRaises(HTTPError) as error:
                urlopen(bad)
            self.assertEqual(error.exception.code, 403)
            good = Request(url + "/api/reviews", data=body,
                           headers={"Content-Type": "application/json", "X-Review-Token": data["token"]})
            with urlopen(good) as response:
                saved = json.load(response)
            self.assertEqual(saved["reviews"][0]["reviewer"], "TEST")
            with urlopen(url) as response:
                self.assertIn(b"Review the sentiment labels", response.read())
            wrong_host = Request(url + "/api/reviews", headers={"Host": "evil.example"})
            with self.assertRaises(HTTPError) as error:
                urlopen(wrong_host)
            self.assertEqual(error.exception.code, 403)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == "__main__":
    unittest.main()
