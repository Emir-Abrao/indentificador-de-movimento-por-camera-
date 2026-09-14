import csv
import sqlite3

import pytest

from people_counter.storage import Store, export_csv
from people_counter.types import Crossing


def event(seq=1, direction="in"):
    return Crossing(seq, 10, direction, 20, 0.66, 0.3, 0.5)


def test_event_retry_snapshot_and_session_persistence(tmp_path):
    path = tmp_path / "counts.db"
    store = Store(path, "door", {"detector": "hog"})
    first_id = store.session_id
    store.record([event()], 20, {"visible": 1})
    store.record([event()], 20, {"visible": 2})
    store.close()
    second = Store(path, "door", {})
    second.record([event()], 20)
    assert first_id != second.session_id
    second.close("interrupted")
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT COUNT(*) FROM crossings").fetchone()[0] == 2
        assert db.execute("SELECT COUNT(*) FROM snapshots").fetchone()[0] == 1
        statuses = db.execute("SELECT status FROM sessions ORDER BY started_at").fetchall()
        assert statuses == [("completed",), ("interrupted",)]


def test_bad_event_rolls_back_whole_batch(tmp_path):
    store = Store(tmp_path / "counts.db", "door", {})
    with pytest.raises(sqlite3.IntegrityError):
        store.record([event(), event(2, "invalid")], 1)
    assert store.connection.execute("SELECT COUNT(*) FROM crossings").fetchone()[0] == 0
    store.close()


def test_export_is_safe_readonly_and_no_overwrite(tmp_path):
    database, output = tmp_path / "counts.db", tmp_path / "events.csv"
    store = Store(database, "=HYPERLINK()", {})
    store.record([event()], 20)
    store.close()
    assert export_csv(database, output) == 1
    with output.open() as file:
        rows = list(csv.DictReader(file))
    assert rows[0]["camera_label"] == "'=HYPERLINK()"
    assert rows[0]["direction"] == "in"
    with pytest.raises(FileExistsError):
        export_csv(database, output)
    with pytest.raises(FileNotFoundError):
        export_csv(tmp_path / "absent.db", tmp_path / "absent.csv")
    assert not (tmp_path / "absent.db").exists()
