from pathlib import Path

from job_classifier.cli import read_csv_rows

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def test_read_raw_positions_row_count():
    rows = read_csv_rows(DATA_DIR / "raw_positions.csv")
    assert len(rows) == 300


def test_read_raw_positions_bom_stripped_from_header():
    rows = read_csv_rows(DATA_DIR / "raw_positions.csv")
    assert "id" in rows[0]
