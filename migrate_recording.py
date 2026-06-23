#!/usr/bin/env python3
"""Migrate an old-format recording.db to new capture.db format.

OpenAdapt v1.0+ expects capture.db with 'capture' and 'events' tables.
Old recordings use recording.db with 'recording', 'action_event', 'screenshot' tables.

This script converts old format → new format so the data can be used with
openadapt-capture Capture.load() and openadapt-ml training pipeline.

Usage:
    python migrate_recording.py --capture ./my-task
    python migrate_recording.py --capture ./my-task --dry-run
"""

import argparse
import json
import os
import shutil
import sqlite3
import sys
from pathlib import Path


def migrate(capture_dir: str, dry_run: bool = False) -> None:
    """Migrate recording.db to capture.db format."""
    capture_path = Path(capture_dir)
    old_db = capture_path / "recording.db"
    new_db = capture_path / "capture.db"

    if not old_db.exists():
        print(f"❌ recording.db not found in {capture_path}")
        sys.exit(1)

    if new_db.exists():
        print(f"⚠️  capture.db already exists in {capture_path}")
        if not dry_run:
            backup = capture_path / "capture.db.bak"
            shutil.copy2(new_db, backup)
            print(f"   Backed up to {backup}")

    # Open old database
    old_conn = sqlite3.connect(str(old_db))
    old_conn.row_factory = sqlite3.Row
    old_cur = old_conn.cursor()

    # Read recording metadata
    old_cur.execute("SELECT * FROM recording LIMIT 1")
    recording = dict(old_cur.fetchone())
    print(f"📋 Recording: {recording.get('task_description', 'N/A')}")
    print(f"   Platform: {recording.get('platform', 'N/A')}")
    print(f"   Screen: {recording.get('monitor_width')}x{recording.get('monitor_height')}")

    # Read action events
    old_cur.execute("SELECT * FROM action_event ORDER BY timestamp")
    action_events = [dict(r) for r in old_cur.fetchall()]
    print(f"   Action events: {len(action_events)}")

    # Read screenshots
    old_cur.execute("SELECT * FROM screenshot ORDER BY timestamp")
    screenshots = [dict(r) for r in old_cur.fetchall()]
    print(f"   Screenshots: {len(screenshots)}")

    if dry_run:
        print("\n🔍 DRY RUN — would create capture.db with above data")
        old_conn.close()
        return

    # Create new capture.db
    if new_db.exists():
        os.remove(new_db)

    new_conn = sqlite3.connect(str(new_db))
    new_cur = new_conn.cursor()

    # Create capture table
    new_cur.execute("""
        CREATE TABLE IF NOT EXISTS capture (
            id TEXT PRIMARY KEY,
            started_at REAL,
            ended_at REAL,
            platform TEXT,
            screen_width INTEGER,
            screen_height INTEGER,
            pixel_ratio REAL DEFAULT 1.0,
            task_description TEXT,
            double_click_interval_seconds REAL,
            double_click_distance_pixels REAL,
            video_start_time REAL,
            audio_start_time REAL,
            metadata JSON
        )
    """)

    # Create events table
    new_cur.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL NOT NULL,
            type TEXT NOT NULL,
            data JSON,
            parent_id INTEGER REFERENCES events(id)
        )
    """)

    # Insert capture metadata
    capture_id = str(recording.get("id", "migrated"))
    started_at = recording.get("timestamp", 0.0)

    # Find end time from last event
    ended_at = started_at
    if action_events:
        ended_at = max(e["timestamp"] for e in action_events)

    new_cur.execute("""
        INSERT INTO capture (id, started_at, ended_at, platform, screen_width,
                            screen_height, pixel_ratio, task_description,
                            double_click_interval_seconds, double_click_distance_pixels,
                            video_start_time, metadata)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        capture_id,
        started_at,
        ended_at,
        recording.get("platform", "darwin"),
        recording.get("monitor_width", 1920),
        recording.get("monitor_height", 1080),
        1.0,  # pixel_ratio - Retina displays have 2.0 but coords are in points
        recording.get("task_description", "migrated recording"),
        recording.get("double_click_interval_seconds", 0.5),
        recording.get("double_click_distance_pixels", 5.0),
        recording.get("video_start_time"),
        json.dumps({"migrated_from": "recording.db"}),
    ))

    # Convert action events to new events format
    event_id = 0
    for ae in action_events:
        event_id += 1
        event_type = _map_event_type(ae)
        event_data = _build_event_data(ae)

        new_cur.execute("""
            INSERT INTO events (id, timestamp, type, data, parent_id)
            VALUES (?, ?, ?, ?, ?)
        """, (
            event_id,
            ae["timestamp"],
            event_type,
            json.dumps(event_data),
            ae.get("parent_id"),
        ))

    # Also insert screenshot events
    for ss in screenshots:
        if ss.get("png_data"):
            event_id += 1
            new_cur.execute("""
                INSERT INTO events (id, timestamp, type, data, parent_id)
                VALUES (?, ?, ?, ?, ?)
            """, (
                event_id,
                ss["timestamp"],
                "screen_frame",
                json.dumps({
                    "png_data": None,  # binary data handled separately
                    "screenshot_id": ss["id"],
                }),
                None,
            ))

    new_conn.commit()
    new_conn.close()
    old_conn.close()

    print(f"\n✅ Successfully migrated to: {new_db}")
    print(f"   Capture ID: {capture_id}")
    print(f"   Events: {event_id}")


def _map_event_type(action_event: dict) -> str:
    """Map old action_event 'name' to new event type string."""
    name = action_event.get("name", "")
    has_mouse = action_event.get("mouse_x") is not None

    if name == "press":
        if has_mouse:
            return "mouse_down"
        return "key_down"
    elif name == "release":
        if has_mouse:
            return "mouse_up"
        return "key_up"
    elif name == "click":
        return "mouse_click"
    elif name == "singleclick":
        return "mouse_click"
    elif name == "doubleclick":
        return "mouse_double_click"
    elif name == "move":
        return "mouse_move"
    elif name == "scroll":
        return "mouse_scroll"
    elif name == "type":
        return "key_type"
    else:
        return name


def _build_event_data(action_event: dict) -> dict:
    """Build event data JSON from old action_event row."""
    data = {}

    # Mouse data
    if action_event.get("mouse_x") is not None:
        data["x"] = action_event["mouse_x"]
        data["y"] = action_event["mouse_y"]
    if action_event.get("mouse_dx") is not None:
        data["dx"] = action_event["mouse_dx"]
        data["dy"] = action_event["mouse_dy"]
    if action_event.get("mouse_button_name") is not None:
        data["button"] = action_event["mouse_button_name"]
    if action_event.get("mouse_pressed") is not None:
        data["pressed"] = bool(action_event["mouse_pressed"])

    # Keyboard data
    if action_event.get("key_name") is not None:
        data["key_name"] = action_event["key_name"]
    if action_event.get("key_char") is not None:
        data["key_char"] = action_event["key_char"]
    if action_event.get("key_vk") is not None:
        data["key_vk"] = action_event["key_vk"]
    if action_event.get("canonical_key_name") is not None:
        data["canonical_key_name"] = action_event["canonical_key_name"]
    if action_event.get("canonical_key_char") is not None:
        data["canonical_key_char"] = action_event["canonical_key_char"]

    # Screenshot reference
    if action_event.get("screenshot_id") is not None:
        data["screenshot_id"] = action_event["screenshot_id"]

    return data


def main():
    parser = argparse.ArgumentParser(
        description="Migrate recording.db to capture.db format.",
    )
    parser.add_argument(
        "--capture", "-c",
        default="./my-task",
        help="Path to the capture directory (default: ./my-task)",
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Preview without creating files",
    )
    args = parser.parse_args()

    migrate(args.capture, args.dry_run)


if __name__ == "__main__":
    main()
