import tempfile
import unittest
from pathlib import Path

from services.history import load_history, save_analysis, summarize_history


class WasteHistoryTests(unittest.TestCase):
    def test_saves_and_summarizes_analysis_results(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "history.sqlite3"
            detections = [
                {"category": "Plastic", "confidence": 0.91, "box": [1, 2, 3, 4]},
                {"category": "Organic waste", "confidence": 0.82, "box": [4, 5, 6, 7]},
            ]
            save_analysis("upload.png", "YOLO · test.pt", detections, database)

            history = load_history(database)
            summary = summarize_history(history)

        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["filename"], "upload.png")
        self.assertEqual(history[0]["category_counts"], {"Plastic": 1, "Organic waste": 1})
        self.assertEqual(summary["analyses"], 1)
        self.assertEqual(summary["detections"], 2)
        self.assertEqual(summary["recyclable"], 1)
        self.assertEqual(summary["recycling_rate"], 50)

    def test_empty_history_has_zero_recycling_rate(self):
        summary = summarize_history([])

        self.assertEqual(summary["analyses"], 0)
        self.assertEqual(summary["detections"], 0)
        self.assertEqual(summary["recycling_rate"], 0)
        self.assertTrue(all(count == 0 for count in summary["categories"].values()))

    def test_missing_database_loads_as_empty(self):
        with tempfile.TemporaryDirectory() as directory:
            history = load_history(Path(directory) / "missing.sqlite3")

        self.assertEqual(history, [])


if __name__ == "__main__":
    unittest.main()
