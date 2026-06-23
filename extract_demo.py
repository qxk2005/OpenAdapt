#!/usr/bin/env python3
"""Extract a demo trajectory text file from an OpenAdapt recording.

This generates a human-readable demo text that can be used with:
    openadapt eval run --agent api-openai --demo ./my-task/demo.txt

Usage:
    python extract_demo.py                        # Extract from my-task
    python extract_demo.py --capture ./my-task    # Specify capture path
    python extract_demo.py --output demo.txt      # Custom output path
"""

import argparse
import sqlite3
import sys
import os


def extract_demo(db_path: str, task_description: str | None = None) -> str:
    """Extract a demo trajectory text from a recording database.

    Args:
        db_path: Path to the recording.db file.
        task_description: Optional override for the task description.

    Returns:
        A formatted demo trajectory string.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get task description from recording table
    if not task_description:
        cursor.execute("SELECT task_description FROM recording LIMIT 1")
        row = cursor.fetchone()
        task_description = row[0] if row else "GUI automation task"

    # Get all action events
    cursor.execute("""
        SELECT id, name, timestamp, mouse_x, mouse_y,
               mouse_button_name, mouse_pressed,
               key_name, key_char,
               canonical_key_name, canonical_key_char
        FROM action_event
        ORDER BY timestamp
    """)
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return f"Task: {task_description}\n\nNo events recorded."

    lines = [f"Task: {task_description}", ""]
    step = 0
    i = 0

    while i < len(rows):
        _id, name, ts, mx, my, btn, pressed, kn, kc, ckn, ckc = rows[i]

        # Detect modifier + key combos (Cmd+Space, Ctrl+C, etc.)
        if name == "press" and kn and kn.lower() in ("cmd", "cmd_l", "cmd_r",
                                                       "ctrl", "ctrl_l", "ctrl_r",
                                                       "alt", "alt_l", "alt_r",
                                                       "shift", "shift_l", "shift_r"):
            modifier = kn.capitalize()
            # Look ahead for the next press event
            j = i + 1
            while j < len(rows) and rows[j][1] == "release":
                j += 1
            if j < len(rows) and rows[j][1] == "press":
                next_kn = rows[j][7]  # key_name
                next_kc = rows[j][8]  # key_char
                key2 = next_kn or next_kc or "?"
                step += 1
                lines.append(f"Step {step}: Press {modifier}+{key2}")
                # Skip until both are released
                i = j + 1
                release_count = 0
                while i < len(rows) and release_count < 2:
                    if rows[i][1] == "release":
                        release_count += 1
                    i += 1
                continue

        # Detect typed text sequences
        if name == "press" and kc and len(kc) == 1:
            typed = kc
            j = i + 1
            while j < len(rows):
                if rows[j][1] == "press" and rows[j][8] and len(rows[j][8]) == 1:
                    typed += rows[j][8]
                    j += 1
                elif rows[j][1] == "release":
                    j += 1
                else:
                    break
            step += 1
            lines.append(f'Step {step}: Type "{typed}"')
            i = j
            continue

        # Special key press (Enter, Backspace, etc.)
        if name == "press" and kn and kn.lower() in ("enter", "return", "tab",
                                                       "backspace", "delete",
                                                       "escape", "space"):
            step += 1
            lines.append(f"Step {step}: Press {kn.capitalize()}")
            # Skip the release
            i += 1
            while i < len(rows) and rows[i][1] == "release":
                i += 1
            continue

        # Mouse click
        if name in ("click", "singleclick") and mx is not None:
            step += 1
            btn_desc = btn or "left"
            lines.append(
                f"Step {step}: {btn_desc.capitalize()} click at coordinates "
                f"({int(mx)}, {int(my)})"
            )
            i += 1
            continue

        # Mouse double click
        if name == "doubleclick" and mx is not None:
            step += 1
            lines.append(
                f"Step {step}: Double-click at coordinates ({int(mx)}, {int(my)})"
            )
            i += 1
            continue

        # Mouse scroll
        if name == "scroll":
            step += 1
            lines.append(f"Step {step}: Scroll at ({int(mx or 0)}, {int(my or 0)})")
            i += 1
            continue

        # Skip release events and unknown events
        i += 1

    lines.append("")
    lines.append("DONE: Task completed successfully.")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Extract a demo trajectory text from an OpenAdapt recording.",
    )
    parser.add_argument(
        "--capture", "-c",
        default="./my-task",
        help="Path to the capture directory (default: ./my-task)",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output file path (default: <capture>/demo.txt)",
    )
    parser.add_argument(
        "--task", "-t",
        default=None,
        help="Override the task description",
    )

    args = parser.parse_args()

    # Locate the database
    db_path = os.path.join(args.capture, "recording.db")
    if not os.path.exists(db_path):
        db_path = os.path.join(args.capture, "capture.db")
    if not os.path.exists(db_path):
        print(f"❌ Error: Database not found in: {args.capture}")
        sys.exit(1)

    # Extract demo
    demo_text = extract_demo(db_path, args.task)

    # Determine output path
    output_path = args.output or os.path.join(args.capture, "demo.txt")

    # Write output
    with open(output_path, "w") as f:
        f.write(demo_text)

    print(f"✅ Demo trajectory extracted to: {output_path}")
    print(f"\n--- Preview ---")
    print(demo_text)
    print(f"--- End ---\n")
    print(f"Usage with OpenAdapt eval:")
    print(f"  openadapt eval run --agent api-openai --demo {output_path}")


if __name__ == "__main__":
    main()
