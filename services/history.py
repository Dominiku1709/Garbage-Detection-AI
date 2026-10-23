import json
import sqlite3
from collections import Counter
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from services.classifier import CATEGORIES

DEFAULT_DATABASE = Path(__file__).resolve().parent.parent / "output" / "waste_history.sqlite3"
RECYCLABLE_CATEGORIES = {"Plastic", "Paper", "Metal", "Glass"}


def _connect(database_path=DEFAULT_DATABASE):
    path = Path(database_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            analyzed_at TEXT NOT NULL,
            filename TEXT NOT NULL,
            model_name TEXT NOT NULL,
            total_detections INTEGER NOT NULL,
            results_json TEXT NOT NULL
        )
        """
    )
    return connection


@contextmanager
def _database(database_path):
    connection = _connect(database_path)
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def save_analysis(filename, model_name, detections, database_path=DEFAULT_DATABASE):
    category_counts = Counter(item["category"] for item in detections)
    record = {
        "categories": dict(category_counts),
        "detections": detections,
    }
    analyzed_at = datetime.now().astimezone().isoformat(timespec="seconds")
    with _database(database_path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO analyses (analyzed_at, filename, model_name, total_detections, results_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (analyzed_at, Path(filename).name, model_name, len(detections), json.dumps(record)),
        )
        return cursor.lastrowid


def load_history(database_path=DEFAULT_DATABASE, limit=500):
    if limit < 1:
        raise ValueError("limit must be greater than zero")
    if not Path(database_path).is_file():
        return []
    with _database(database_path) as connection:
        rows = connection.execute(
            """
            SELECT id, analyzed_at, filename, model_name, total_detections, results_json
            FROM analyses
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    history = []
    for row in rows:
        result = json.loads(row["results_json"])
        history.append({
            "id": row["id"],
            "analyzed_at": row["analyzed_at"],
            "filename": row["filename"],
            "model_name": row["model_name"],
            "total_detections": row["total_detections"],
            "detections": result["detections"],
            "category_counts": result["categories"],
        })
    return history


def summarize_history(history):
    categories = Counter({category: 0 for category in CATEGORIES})
    analyses = len(history)
    detections = 0
    for record in history:
        detections += record["total_detections"]
        categories.update(record["category_counts"])
    recyclable = sum(categories[category] for category in RECYCLABLE_CATEGORIES)
    return {
        "analyses": analyses,
        "detections": detections,
        "recyclable": recyclable,
        "recycling_rate": (recyclable / detections * 100) if detections else 0.0,
        "categories": dict(categories),
    }
