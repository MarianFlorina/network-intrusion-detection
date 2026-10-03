from __future__ import annotations

from src import db


def test_insert_and_fetch(temp_db):
    rows = [
        {"batch_id": "b1", "source": "test", "duration": 1.0, "protocol": "tcp",
         "service": "http", "flag": "SF", "src_bytes": 100.0, "dst_bytes": 200.0,
         "count": 5.0, "srv_count": 3.0, "same_srv_rate": 0.5, "diff_srv_rate": 0.2,
         "src_port_entropy": 2.0, "packet_rate": 50.0, "connection_duration_std": 1.0,
         "failed_logins": 0.0, "bytes_per_packet": 40.0, "attack_type": "normal"},
        {"batch_id": "b1", "source": "test", "duration": 2.0, "protocol": "tcp",
         "service": "ssh", "flag": "REJ", "src_bytes": 50.0, "dst_bytes": 60.0,
         "count": 40.0, "srv_count": 30.0, "same_srv_rate": 0.9, "diff_srv_rate": 0.7,
         "src_port_entropy": 5.0, "packet_rate": 300.0, "connection_duration_std": 2.0,
         "failed_logins": 5.0, "bytes_per_packet": 30.0, "attack_type": "brute_force"},
    ]
    import pandas as pd

    inserted = db.insert_dataframe(pd.DataFrame(rows), "processed_records")
    assert inserted == 2

    fetched = db.fetch_dataframe("SELECT * FROM processed_records WHERE batch_id='b1'")
    assert len(fetched) == 2
    assert set(fetched["attack_type"]) == {"normal", "brute_force"}


def test_retraining_event_roundtrip(temp_db):
    event_id = db.insert_retraining_event(
        trigger="drift",
        old_model_version="3",
        new_model_version="4",
        new_run_id="abc123",
        new_f1=0.961,
        old_f1=0.955,
        promoted=True,
        notes="integration test",
    )
    assert event_id is not None
    df = db.fetch_dataframe(f"SELECT * FROM retraining_events WHERE id={event_id}")
    assert df.iloc[0]["promoted"] in (True, 1)
