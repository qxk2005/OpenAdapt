#!/usr/bin/env python3
"""Run the fine-tuned VL model to reproduce recorded GUI operations.

This script:
1. Loads the LoRA-finetuned Qwen2-VL model from training_output/final
2. Takes a screenshot of the current screen
3. Asks the model to predict the next action
4. Executes the action (click/type) using pynput
5. Repeats until the model outputs DONE or max steps reached

Usage:
    python run_model.py                                    # Default settings
    python run_model.py --checkpoint training_output/final # Specify checkpoint
    python run_model.py --max-steps 20                     # More steps
    python run_model.py --delay 5                          # 5s countdown
    python run_model.py --dry-run                          # Preview without executing

Requirements (OAT environment):
    pip install pynput Pillow
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# ── Prompt matching the training format ──────────────────────────────────────
SYSTEM_PROMPT = """You are a GUI automation agent. Given a screenshot and a user goal, predict the single next action.

COORDINATE SYSTEM:
- x=0.0 is the LEFT edge, x=1.0 is the RIGHT edge
- y=0.0 is the TOP edge, y=1.0 is the BOTTOM edge
- To click the CENTER of an element, estimate its center position as a fraction of screen width/height
- Example: An element in the middle of the screen would be approximately x=0.5, y=0.5

ALLOWED ACTIONS (use exactly this format):
- CLICK(x=0.XX, y=0.XX)  → click at normalized coordinates
- TYPE(text="...")     → type text into the currently focused field
- WAIT()                 → wait for UI to update
- DONE()                 → task is complete

RESPONSE FORMAT (required):
Thought: [Brief reasoning: what element to interact with and why]
Action: [Exactly one action, e.g., CLICK(x=0.35, y=0.42)]

IMPORTANT: Output coordinates with 2 decimal places. Estimate the center of target elements."""


def take_screenshot() -> "PIL.Image.Image":
    """Take a screenshot on macOS and return as PIL Image."""
    from PIL import Image

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        tmp_path = f.name

    subprocess.run(
        ["screencapture", "-x", "-C", tmp_path],
        check=True,
        capture_output=True,
    )
    img = Image.open(tmp_path)
    os.unlink(tmp_path)
    return img


def parse_action(response: str) -> dict:
    """Parse model response into an action dict.

    Expected format:
        Thought: ...
        Action: CLICK(x=0.35, y=0.42)
        or
        Action: TYPE(text="hello")
        or
        Action: DONE()
    """
    # Extract the Action line
    action_match = re.search(r"Action:\s*(.+)", response, re.IGNORECASE)
    if not action_match:
        return {"type": "unknown", "raw": response}

    action_str = action_match.group(1).strip()

    # Parse CLICK(x=..., y=...)
    click_match = re.search(r"CLICK\s*\(\s*x\s*=\s*([\d.]+)\s*,\s*y\s*=\s*([\d.]+)\s*\)", action_str, re.IGNORECASE)
    if click_match:
        try:
            x = float(click_match.group(1))
            y = float(click_match.group(2))
            if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
                return {"type": "click", "x": x, "y": y}
            else:
                return {"type": "unknown", "raw": f"Coordinates out of range: x={x}, y={y}"}
        except ValueError:
            return {"type": "unknown", "raw": f"Invalid coordinates in: {action_str}"}

    # Parse TYPE(text="...")
    type_match = re.search(r'TYPE\s*\(\s*text\s*=\s*["\'](.*)["\']\s*\)', action_str, re.IGNORECASE)
    if type_match:
        return {
            "type": "type",
            "text": type_match.group(1),
        }

    # Parse WAIT()
    if re.search(r"WAIT\s*\(\s*\)", action_str, re.IGNORECASE):
        return {"type": "wait"}

    # Parse DONE()
    if re.search(r"DONE\s*\(\s*\)", action_str, re.IGNORECASE):
        return {"type": "done"}

    return {"type": "unknown", "raw": action_str}


def execute_action(action: dict, screen_width: int, screen_height: int, dry_run: bool = False) -> None:
    """Execute a parsed action using pynput."""
    if action["type"] == "click":
        abs_x = int(action["x"] * screen_width)
        abs_y = int(action["y"] * screen_height)
        print(f"  🖱️  Click at ({abs_x}, {abs_y}) [normalized: ({action['x']:.2f}, {action['y']:.2f})]")

        if not dry_run:
            from pynput.mouse import Button, Controller as MouseController
            mouse = MouseController()
            mouse.position = (abs_x, abs_y)
            time.sleep(0.1)
            mouse.click(Button.left, 1)

    elif action["type"] == "type":
        text = action["text"]
        print(f"  ⌨️  Type: {text!r}")

        if not dry_run and text:
            from pynput.keyboard import Controller as KBController
            kb = KBController()
            for char in text:
                kb.press(char)
                kb.release(char)
                time.sleep(0.02)

    elif action["type"] == "wait":
        print(f"  ⏳ Wait 1s")
        if not dry_run:
            time.sleep(1.0)

    elif action["type"] == "done":
        print(f"  ✅ Model signals: DONE")

    else:
        print(f"  ❓ Unknown action: {action}")


def load_model(checkpoint_path: str):
    """Load the fine-tuned model with LoRA adapter."""
    import torch
    from peft import PeftModel
    from transformers import AutoProcessor

    print(f"📂 Loading checkpoint from: {checkpoint_path}")

    # Read adapter config to get base model name
    import json
    adapter_config_path = Path(checkpoint_path) / "adapter_config.json"
    with open(adapter_config_path) as f:
        adapter_config = json.load(f)
    base_model_name = adapter_config["base_model_name_or_path"]
    print(f"   Base model: {base_model_name}")

    # Load base model
    load_kwargs = dict(
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    base_model = None
    if "Qwen2-VL" in base_model_name or "qwen2-vl" in base_model_name.lower():
        try:
            from transformers import Qwen2VLForConditionalGeneration
            print("   Loading with Qwen2VLForConditionalGeneration...")
            base_model = Qwen2VLForConditionalGeneration.from_pretrained(base_model_name, **load_kwargs)
        except Exception as e:
            print(f"   ⚠ Qwen2VLForConditionalGeneration failed: {e}")

    if base_model is None:
        try:
            from transformers import AutoModelForImageTextToText
            print("   Loading with AutoModelForImageTextToText...")
            base_model = AutoModelForImageTextToText.from_pretrained(base_model_name, **load_kwargs)
        except Exception as e:
            from transformers import AutoModelForVision2Seq
            print(f"   Loading with AutoModelForVision2Seq...")
            base_model = AutoModelForVision2Seq.from_pretrained(base_model_name, **load_kwargs)

    # Load LoRA adapter
    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    model.eval()

    # Load processor
    processor = AutoProcessor.from_pretrained(checkpoint_path, trust_remote_code=True)

    print(f"   ✅ Model loaded with LoRA adapter")
    return model, processor


def predict_next_action(model, processor, screenshot, task: str, step: int,
                        total_steps: int, action_history: list[str] | None = None) -> str:
    """Use the model to predict the next action given a screenshot.

    Args:
        action_history: List of action strings from previous steps,
            e.g. ['TYPE(text="")', 'TYPE(text="ji")', 'CLICK(x=0.29, y=0.34)']
            This MUST be included to match the training data format.
    """
    import torch

    # Resize screenshot to match training size
    MAX_WIDTH = 1280
    if screenshot.width > MAX_WIDTH:
        ratio = MAX_WIDTH / screenshot.width
        new_h = int(screenshot.height * ratio)
        screenshot = screenshot.resize((MAX_WIDTH, new_h))

    # Build conversation in the EXACT same format as training data
    # Training samples use this structure:
    #   Goal: <task>
    #   ACTIONS COMPLETED SO FAR:
    #     1. TYPE(text="")
    #     2. TYPE(text="ji")
    #   This is step N of M. Look at the screenshot ...
    if not action_history:
        user_text = (
            f"Goal: {task}\n\n"
            f"This is step {step} of {total_steps} (no actions completed yet). "
            f"Look at the screenshot and determine the NEXT action.\n\n"
            f"Thought: [what element to interact with and why]\n"
            f'Action: [CLICK(x=..., y=...) or TYPE(text="...") or WAIT() or DONE()]'
        )
    else:
        history_lines = "\n".join(
            f"  {i+1}. {a}" for i, a in enumerate(action_history)
        )
        user_text = (
            f"Goal: {task}\n\n"
            f"ACTIONS COMPLETED SO FAR:\n{history_lines}\n\n"
            f"This is step {step} of {total_steps}. "
            f"Look at the screenshot and determine the NEXT action.\n\n"
            f"Thought: [what element to interact with and why]\n"
            f'Action: [CLICK(x=..., y=...) or TYPE(text="...") or WAIT() or DONE()]'
        )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": user_text},
        ]},
    ]

    # Apply chat template
    text = processor.apply_chat_template(messages, add_generation_prompt=True)

    # Process inputs
    inputs = processor(
        text=[text],
        images=[screenshot],
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Generate
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            temperature=1.0,
        )

    # Decode only the new tokens
    input_len = inputs["input_ids"].shape[1]
    generated_ids = output_ids[0][input_len:]
    response = processor.decode(generated_ids, skip_special_tokens=True)

    return response


def main():
    parser = argparse.ArgumentParser(
        description="Run fine-tuned VL model to reproduce recorded GUI operations.",
    )
    parser.add_argument(
        "--checkpoint", "-c",
        default="training_output/final",
        help="Path to the LoRA checkpoint (default: training_output/final)",
    )
    parser.add_argument(
        "--task", "-t",
        default="my-task",
        help="Task description/goal (default: my-task)",
    )
    parser.add_argument(
        "--max-steps", "-s",
        type=int, default=15,
        help="Maximum number of steps (default: 15)",
    )
    parser.add_argument(
        "--delay", "-d",
        type=float, default=5.0,
        help="Seconds to wait before starting (default: 5.0)",
    )
    parser.add_argument(
        "--action-delay",
        type=float, default=1.0,
        help="Seconds to wait between actions (default: 1.0)",
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Preview actions without executing them",
    )

    args = parser.parse_args()

    # Load model
    model, processor = load_model(args.checkpoint)

    # Get screen size
    result = subprocess.run(
        ["python3", "-c", "from AppKit import NSScreen; s=NSScreen.mainScreen().frame(); print(f'{int(s.size.width)},{int(s.size.height)}')"],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        screen_width, screen_height = map(int, result.stdout.strip().split(","))
    else:
        screen_width, screen_height = 2560, 1440  # fallback
    print(f"🖥️  Screen: {screen_width}x{screen_height}")

    # Countdown
    print(f"\n{'='*60}")
    print(f"  Task: {args.task}")
    print(f"  Max steps: {args.max_steps}")
    mode = "DRY RUN" if args.dry_run else "LIVE"
    print(f"  Mode: {mode}")
    print(f"{'='*60}")

    if not args.dry_run:
        print(f"\n⏳ Starting in {args.delay:.0f} seconds...")
        print("   Switch to the target window now!")
        print("   Press Ctrl+C to abort.\n")
        try:
            for i in range(int(args.delay), 0, -1):
                print(f"   {i}...", flush=True)
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n❌ Aborted by user.")
            return

    # Agent loop
    print(f"\n🚀 Starting agent loop...")
    action_history: list[str] = []  # Track actions in training data format

    for step in range(1, args.max_steps + 1):
        print(f"\n--- Step {step}/{args.max_steps} ---")

        # Take screenshot
        print("  📸 Taking screenshot...")
        screenshot = take_screenshot()
        print(f"     Size: {screenshot.width}x{screenshot.height}")

        # Predict next action (with action history matching training format)
        print("  🧠 Predicting next action...")
        response = predict_next_action(
            model, processor, screenshot,
            task=args.task, step=step, total_steps=args.max_steps,
            action_history=action_history if action_history else None,
        )
        print(f"  📝 Model response: {response}")

        # Parse action
        action = parse_action(response)
        print(f"  🎯 Parsed: {action}")

        # Convert parsed action to the string format used in training data
        if action["type"] == "click":
            action_str = f'CLICK(x={action["x"]:.2f}, y={action["y"]:.2f})'
        elif action["type"] == "type":
            action_str = f'TYPE(text="{action.get("text", "")}")'
        elif action["type"] == "wait":
            action_str = "WAIT()"
        elif action["type"] == "done":
            action_str = "DONE()"
        else:
            action_str = f'TYPE(text="")'  # fallback for unknown
        action_history.append(action_str)

        # Check for completion
        if action["type"] == "done":
            execute_action(action, screen_width, screen_height, args.dry_run)
            break

        # Execute action
        execute_action(action, screen_width, screen_height, args.dry_run)

        # Wait for UI to settle
        if not args.dry_run:
            time.sleep(args.action_delay)

    print(f"\n{'='*60}")
    print(f"  Agent finished after {step} steps")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
