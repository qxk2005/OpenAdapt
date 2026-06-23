#!/usr/bin/env python3
"""Replay a captured OpenAdapt recording on macOS.

Reads action events from a recording.db and replays them using pynput,
preserving the original timing between events.

Usage:
    python replay_capture.py                          # Replay my-task at 1x speed
    python replay_capture.py --capture ./my-task      # Specify capture path
    python replay_capture.py --speed 2.0              # 2x speed
    python replay_capture.py --speed 0.5              # Half speed (slower)
    python replay_capture.py --dry-run                # Preview without executing

Requirements:
    pip install pynput

Note:
    macOS requires Accessibility permissions for your terminal.
    Go to: System Settings → Privacy & Security → Accessibility → enable Terminal/iTerm2
"""

import argparse
import sqlite3
import sys
import time
from dataclasses import dataclass


@dataclass
class ActionEvent:
    """Represents a single recorded action event."""
    id: int
    name: str
    timestamp: float
    mouse_x: float | None
    mouse_y: float | None
    mouse_button_name: str | None
    mouse_pressed: bool | None
    key_name: str | None
    key_char: str | None
    canonical_key_name: str | None
    canonical_key_char: str | None


def load_events(db_path: str) -> list[ActionEvent]:
    """Load action events from the recording database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, name, timestamp, mouse_x, mouse_y,
               mouse_button_name, mouse_pressed,
               key_name, key_char,
               canonical_key_name, canonical_key_char
        FROM action_event
        ORDER BY timestamp
    """)
    events = []
    for row in cursor.fetchall():
        events.append(ActionEvent(
            id=row[0], name=row[1], timestamp=row[2],
            mouse_x=row[3], mouse_y=row[4],
            mouse_button_name=row[5],
            mouse_pressed=bool(row[6]) if row[6] is not None else None,
            key_name=row[7], key_char=row[8],
            canonical_key_name=row[9], canonical_key_char=row[10],
        ))
    conn.close()
    return events


# pynput key name → Key object mapping
KEY_NAME_MAP = {
    "cmd": "cmd",
    "cmd_l": "cmd_l",
    "cmd_r": "cmd_r",
    "ctrl": "ctrl",
    "ctrl_l": "ctrl_l",
    "ctrl_r": "ctrl_r",
    "alt": "alt",
    "alt_l": "alt_l",
    "alt_r": "alt_r",
    "option": "alt",
    "shift": "shift",
    "shift_l": "shift_l",
    "shift_r": "shift_r",
    "enter": "enter",
    "return": "enter",
    "tab": "tab",
    "space": "space",
    "backspace": "backspace",
    "delete": "delete",
    "escape": "esc",
    "esc": "esc",
    "up": "up",
    "down": "down",
    "left": "left",
    "right": "right",
    "home": "home",
    "end": "end",
    "page_up": "page_up",
    "page_down": "page_down",
    "caps_lock": "caps_lock",
    "f1": "f1", "f2": "f2", "f3": "f3", "f4": "f4",
    "f5": "f5", "f6": "f6", "f7": "f7", "f8": "f8",
    "f9": "f9", "f10": "f10", "f11": "f11", "f12": "f12",
}


def resolve_key(event: ActionEvent):
    """Resolve an ActionEvent to a pynput key object."""
    from pynput.keyboard import Key, KeyCode

    # Try key_name first (special keys)
    key_name = event.key_name or event.canonical_key_name
    if key_name:
        mapped = KEY_NAME_MAP.get(key_name.lower())
        if mapped:
            try:
                return getattr(Key, mapped)
            except AttributeError:
                pass
        # Unknown special key, try as character
        if len(key_name) == 1:
            return KeyCode.from_char(key_name)

    # Use key_char (regular characters)
    key_char = event.key_char or event.canonical_key_char
    if key_char and len(key_char) == 1:
        return KeyCode.from_char(key_char)

    return None


def describe_event(event: ActionEvent) -> str:
    """Generate a human-readable description of an event."""
    if event.name in ("press", "release"):
        key_name = event.key_name or event.canonical_key_name
        key_char = event.key_char or event.canonical_key_char
        key_desc = key_name or repr(key_char) or "?"
        return f"Key {event.name}: {key_desc}"
    elif event.name in ("click", "singleclick", "doubleclick"):
        return (f"Mouse {event.name} at ({event.mouse_x:.0f}, {event.mouse_y:.0f}) "
                f"btn={event.mouse_button_name} pressed={event.mouse_pressed}")
    elif event.name == "move":
        return f"Mouse move to ({event.mouse_x:.0f}, {event.mouse_y:.0f})"
    elif event.name == "scroll":
        return f"Mouse scroll at ({event.mouse_x:.0f}, {event.mouse_y:.0f})"
    else:
        return f"{event.name}: {event}"


def replay_events(
    events: list[ActionEvent],
    speed: float = 1.0,
    dry_run: bool = False,
    delay_before_start: float = 3.0,
) -> None:
    """Replay a list of action events.

    Args:
        events: List of ActionEvents to replay.
        speed: Speed multiplier (2.0 = twice as fast).
        dry_run: If True, only print actions without executing.
        delay_before_start: Seconds to wait before starting replay.
    """
    if not events:
        print("No events to replay.")
        return

    if not dry_run:
        from pynput.keyboard import Controller as KBController
        from pynput.mouse import Button, Controller as MouseController

        kb = KBController()
        mouse = MouseController()

    print(f"\n{'='*60}")
    print(f"  Replay: {len(events)} events at {speed}x speed")
    print(f"{'='*60}")

    if not dry_run:
        print(f"\n⏳ Starting in {delay_before_start:.0f} seconds...")
        print("   Switch to the target window now!")
        print("   Press Ctrl+C to abort.\n")
        try:
            for i in range(int(delay_before_start), 0, -1):
                print(f"   {i}...", flush=True)
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n❌ Aborted by user.")
            return
    else:
        print("\n🔍 DRY RUN — showing actions without executing:\n")

    prev_ts = events[0].timestamp
    played = 0

    try:
        for i, event in enumerate(events):
            # Calculate and apply delay
            dt = (event.timestamp - prev_ts) / speed
            if dt > 0 and not dry_run:
                # Cap max delay to 5 seconds to avoid long waits
                dt = min(dt, 5.0)
                time.sleep(dt)
            prev_ts = event.timestamp

            desc = describe_event(event)

            if dry_run:
                print(f"  [{i+1:3d}/{len(events)}] {desc}")
                continue

            # Execute the event
            if event.name == "press":
                key = resolve_key(event)
                if key:
                    kb.press(key)
                    played += 1
                    print(f"  ▶ [{i+1:3d}] {desc}")
                else:
                    print(f"  ⚠ [{i+1:3d}] Skip (unresolved key): {desc}")

            elif event.name == "release":
                key = resolve_key(event)
                if key:
                    kb.release(key)
                    played += 1
                    # Don't print release events to reduce noise

            elif event.name == "click":
                if event.mouse_x is not None and event.mouse_y is not None:
                    mouse.position = (int(event.mouse_x), int(event.mouse_y))
                    btn_name = event.mouse_button_name or "left"
                    btn = Button[btn_name]
                    if event.mouse_pressed:
                        mouse.press(btn)
                    else:
                        mouse.release(btn)
                    played += 1
                    print(f"  ▶ [{i+1:3d}] {desc}")

            elif event.name == "singleclick":
                if event.mouse_x is not None and event.mouse_y is not None:
                    mouse.position = (int(event.mouse_x), int(event.mouse_y))
                    btn_name = event.mouse_button_name or "left"
                    btn = Button[btn_name]
                    mouse.click(btn, 1)
                    played += 1
                    print(f"  ▶ [{i+1:3d}] {desc}")

            elif event.name == "doubleclick":
                if event.mouse_x is not None and event.mouse_y is not None:
                    mouse.position = (int(event.mouse_x), int(event.mouse_y))
                    btn_name = event.mouse_button_name or "left"
                    btn = Button[btn_name]
                    mouse.click(btn, 2)
                    played += 1
                    print(f"  ▶ [{i+1:3d}] {desc}")

            elif event.name == "move":
                if event.mouse_x is not None and event.mouse_y is not None:
                    mouse.position = (int(event.mouse_x), int(event.mouse_y))
                    played += 1

            elif event.name == "scroll":
                if event.mouse_x is not None and event.mouse_y is not None:
                    mouse.position = (int(event.mouse_x), int(event.mouse_y))
                    dx = event.mouse_x if hasattr(event, "mouse_dx") else 0
                    dy = event.mouse_y if hasattr(event, "mouse_dy") else 0
                    mouse.scroll(int(dx), int(dy))
                    played += 1
                    print(f"  ▶ [{i+1:3d}] {desc}")

            else:
                print(f"  ⚠ [{i+1:3d}] Unknown event type: {event.name}")

    except KeyboardInterrupt:
        print(f"\n\n❌ Replay interrupted at event {i+1}/{len(events)}")
        return

    print(f"\n{'='*60}")
    if dry_run:
        print(f"  ✅ Dry run complete: {len(events)} events listed")
    else:
        print(f"  ✅ Replay complete: {played} events executed")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Replay a captured OpenAdapt recording.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python replay_capture.py                        # Replay my-task at 1x speed
  python replay_capture.py --capture ./my-task    # Specify capture path
  python replay_capture.py --speed 2.0            # 2x speed
  python replay_capture.py --dry-run              # Preview without executing
  python replay_capture.py --delay 5              # 5 second countdown
        """,
    )
    parser.add_argument(
        "--capture", "-c",
        default="./my-task",
        help="Path to the capture directory (default: ./my-task)",
    )
    parser.add_argument(
        "--speed", "-s",
        type=float, default=1.0,
        help="Playback speed multiplier (default: 1.0)",
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Preview events without executing them",
    )
    parser.add_argument(
        "--delay", "-d",
        type=float, default=3.0,
        help="Seconds to wait before starting (default: 3.0)",
    )

    args = parser.parse_args()

    # Locate the database
    import os
    db_path = os.path.join(args.capture, "recording.db")
    if not os.path.exists(db_path):
        # Also try capture.db (newer format)
        db_path = os.path.join(args.capture, "capture.db")
    if not os.path.exists(db_path):
        print(f"❌ Error: Recording database not found in: {args.capture}")
        print(f"   Looked for: recording.db and capture.db")
        sys.exit(1)

    print(f"📂 Loading capture from: {db_path}")
    events = load_events(db_path)
    print(f"📊 Found {len(events)} action events")

    if not events:
        print("❌ No events found in database.")
        sys.exit(1)

    # Show a summary
    duration = events[-1].timestamp - events[0].timestamp
    print(f"⏱  Original duration: {duration:.1f}s")
    print(f"⚡ Replay duration: ~{duration/args.speed:.1f}s (at {args.speed}x speed)")

    replay_events(
        events,
        speed=args.speed,
        dry_run=args.dry_run,
        delay_before_start=args.delay,
    )


if __name__ == "__main__":
    main()
