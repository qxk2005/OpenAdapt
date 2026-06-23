#!/usr/bin/env python3
"""Train a VL model on an OpenAdapt capture recording.

This script bypasses the CLI and openadapt-ml's model loader to work around:
1. CLI/ML version mismatches (capture_to_episode 'goal' param)
2. openadapt-ml 0.2.0 uses AutoModelForCausalLM for VL models (wrong)
3. Default model name Qwen2.5-VL-2B doesn't exist (minimum is 3B)

Usage:
    python train_local.py                                       # Default (Qwen2-VL-2B)
    python train_local.py --model Qwen/Qwen2.5-VL-3B-Instruct  # Qwen2.5-VL-3B
    python train_local.py --epochs 10                           # More epochs
    python train_local.py --dry-run                             # Preview only

Requirements (OAT environment):
    pip install trl datasets peft accelerate
"""

import argparse
import os
import sys
from pathlib import Path


def _load_vl_model_for_training(model_name: str, lora_r: int, lora_alpha: int,
                                  lora_dropout: float, load_in_4bit: bool):
    """Load a VL model with LoRA for training (fixes openadapt-ml 0.2.0 bug).

    openadapt-ml 0.2.0's _load_standard_model uses AutoModelForCausalLM,
    which is wrong for vision-language models. This function uses the correct
    AutoModelForVision2Seq / AutoModelForImageTextToText.
    """
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoProcessor

    load_kwargs = dict(
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # 4-bit quantization (CUDA only)
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    # Try multiple model classes for VL models
    model = None

    # Approach 1: Specific Qwen2VL class (most reliable)
    if "Qwen2-VL" in model_name or "qwen2-vl" in model_name.lower():
        try:
            from transformers import Qwen2VLForConditionalGeneration
            print("   Loading with Qwen2VLForConditionalGeneration...")
            model = Qwen2VLForConditionalGeneration.from_pretrained(model_name, **load_kwargs)
            print("   ✅ Loaded with Qwen2VLForConditionalGeneration")
        except Exception as e:
            print(f"   ⚠ Qwen2VLForConditionalGeneration failed: {e}")

    # Approach 2: AutoModelForImageTextToText (transformers 5.x)
    if model is None:
        try:
            from transformers import AutoModelForImageTextToText
            print("   Loading with AutoModelForImageTextToText...")
            model = AutoModelForImageTextToText.from_pretrained(model_name, **load_kwargs)
            print("   ✅ Loaded with AutoModelForImageTextToText")
        except Exception as e:
            print(f"   ⚠ AutoModelForImageTextToText failed: {e}")

    # Approach 3: AutoModelForVision2Seq (transformers 4.x)
    if model is None:
        try:
            from transformers import AutoModelForVision2Seq
            print("   Loading with AutoModelForVision2Seq...")
            model = AutoModelForVision2Seq.from_pretrained(model_name, **load_kwargs)
            print("   ✅ Loaded with AutoModelForVision2Seq")
        except Exception as e:
            print(f"   ⚠ AutoModelForVision2Seq failed: {e}")

    if model is None:
        print("❌ Could not load model with any known VL model class")
        sys.exit(1)

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    # Apply LoRA
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    return model, processor


def main():
    parser = argparse.ArgumentParser(
        description="Train a VL model on an OpenAdapt capture recording.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--capture", "-c",
        default="./my-task",
        help="Path to the capture directory (default: ./my-task)",
    )
    parser.add_argument(
        "--model", "-m",
        default="Qwen/Qwen3-VL-2B-Instruct",
        help="Model name from HuggingFace (default: Qwen/Qwen3-VL-2B-Instruct)",
    )
    parser.add_argument(
        "--output", "-o",
        default="training_output",
        help="Output directory for checkpoints (default: training_output)",
    )
    parser.add_argument(
        "--epochs", "-e",
        type=int, default=40,
        help="Number of training epochs (default: 40)",
    )
    parser.add_argument(
        "--lr",
        type=float, default=2e-4,
        help="Learning rate (default: 2e-4)",
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int, default=1,
        help="Batch size (default: 1)",
    )
    parser.add_argument(
        "--lora-r",
        type=int, default=32,
        help="LoRA rank (default: 32)",
    )
    parser.add_argument(
        "--4bit",
        action="store_true",
        dest="load_4bit",
        help="Enable 4-bit quantization (requires CUDA + bitsandbytes)",
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Only load data and show what would be trained, don't actually train",
    )

    args = parser.parse_args()

    # Auto-detect device and set constraints
    os.environ["OPENADAPT_DISABLE_UNSLOTH"] = "1"  # Always disable on Mac

    import torch
    if not torch.cuda.is_available():
        device_info = "MPS (Apple Silicon)" if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()) else "CPU"
        print(f"🖥️  Device: {device_info}")
        if args.load_4bit:
            print("⚠️  4-bit quantization disabled (bitsandbytes requires CUDA)")
            args.load_4bit = False
    else:
        print(f"🖥️  Device: CUDA ({torch.cuda.get_device_name(0)})")

    # Step 1: Load capture data
    print(f"\n{'='*60}")
    print(f"  Step 1: Loading capture data")
    print(f"{'='*60}")

    from openadapt_ml.ingest.capture import capture_to_episode

    capture_path = Path(args.capture)
    if not capture_path.exists():
        print(f"❌ Capture directory not found: {capture_path}")
        sys.exit(1)

    episode = capture_to_episode(str(capture_path))
    print(f"✅ Loaded episode: {episode.episode_id}")
    print(f"   Instruction: {episode.instruction}")
    print(f"   Raw steps: {len(episode.steps)}")

    # ─── Step 1b: Clean episode data ─────────────────────────────────
    # The capture_to_episode pipeline has known data quality issues:
    #  1. Keyboard shortcuts (Cmd+Space, Shift, Enter) become TYPE(text="")
    #  2. Typed words are split into individual characters
    # Both problems severely degrade training quality.
    print(f"\n   🧹 Cleaning episode data...")

    from openadapt_ml.schema import Action, ActionType, Step, Observation

    cleaned_steps = []
    i = 0
    raw_steps = episode.steps
    removed_empty = 0
    merged_chars = 0

    while i < len(raw_steps):
        step = raw_steps[i]
        action = step.action

        # 1. Remove empty TYPE(text="") actions (Shift, Cmd+Space, Enter)
        if action.type == ActionType.TYPE and (not action.text or action.text.strip() == ""):
            removed_empty += 1
            i += 1
            continue

        # 2. Merge consecutive single-char TYPE actions into one word
        if action.type == ActionType.TYPE and action.text and len(action.text) <= 2:
            merged_text = action.text
            base_observation = step.observation  # Keep the screenshot from the FIRST char
            j = i + 1
            while j < len(raw_steps):
                next_step = raw_steps[j]
                next_action = next_step.action
                if next_action.type == ActionType.TYPE and next_action.text and len(next_action.text) <= 2:
                    merged_text += next_action.text
                    merged_chars += 1
                    j += 1
                else:
                    break
            # Only merge if we actually combined multiple chars
            if j > i + 1:
                merged_action = Action(
                    type=ActionType.TYPE,
                    text=merged_text,
                )
                merged_step = Step(
                    action=merged_action,
                    observation=base_observation,
                    step_index=len(cleaned_steps),
                )
                cleaned_steps.append(merged_step)
                i = j
                continue

        # 3. Keep all other steps (CLICK, DONE, etc.) as-is
        step.step_index = len(cleaned_steps)
        cleaned_steps.append(step)
        i += 1

    # Ensure the last step is DONE
    if not cleaned_steps or cleaned_steps[-1].action.type != ActionType.DONE:
        done_step = Step(
            action=Action(type=ActionType.DONE),
            observation=cleaned_steps[-1].observation if cleaned_steps else raw_steps[-1].observation,
            step_index=len(cleaned_steps),
        )
        cleaned_steps.append(done_step)

    episode.steps = cleaned_steps

    print(f"   ✅ Removed {removed_empty} empty TYPE actions")
    print(f"   ✅ Merged {merged_chars} single-char TYPE actions")
    print(f"   ✅ Cleaned steps: {len(episode.steps)} (was {len(raw_steps)})")
    print()
    for i, step in enumerate(episode.steps):
        text_info = f", text={step.action.text!r}" if step.action.text else ""
        coord_info = ""
        if step.action.normalized_coordinates:
            x, y = step.action.normalized_coordinates
            coord_info = f", coords=({x:.3f}, {y:.3f})"
        print(f"     Step {i}: {step.action.type.value}{text_info}{coord_info}")

    # Step 2: Convert to training samples
    print(f"\n{'='*60}")
    print(f"  Step 2: Preparing training data")
    print(f"{'='*60}")

    from openadapt_ml.datasets.next_action import build_next_action_sft_samples
    from openadapt_ml.training.trl_trainer import _convert_samples_to_trl_format

    raw_samples = build_next_action_sft_samples([episode])
    print(f"   Generated {len(raw_samples)} SFT samples")

    trl_samples = _convert_samples_to_trl_format(raw_samples, base_path=Path("."))
    print(f"   Loaded {len(trl_samples)} samples with images")

    if not trl_samples:
        print("❌ No valid training samples! Check that screenshots exist.")
        sys.exit(1)

    # Step 3: Configuration summary
    print(f"\n{'='*60}")
    print(f"  Step 3: Training configuration")
    print(f"{'='*60}")
    print(f"   Model: {args.model}")
    print(f"   4-bit: {args.load_4bit}")
    print(f"   LoRA rank: {args.lora_r}")
    print(f"   Epochs: {args.epochs}")
    print(f"   Learning rate: {args.lr}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Gradient accumulation: 4")
    print(f"   Training samples: {len(trl_samples)}")
    print(f"   Output: {args.output}")

    if args.dry_run:
        print(f"\n🔍 DRY RUN — would train {args.model} on {len(trl_samples)} samples")
        print(f"   Run without --dry-run to start training")
        return

    # Step 4: Load model (our fixed version)
    print(f"\n{'='*60}")
    print(f"  Step 4: Loading model")
    print(f"{'='*60}")

    model, processor = _load_vl_model_for_training(
        model_name=args.model,
        lora_r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        lora_dropout=0.05,
        load_in_4bit=args.load_4bit,
    )

    # Step 4b: Resize images to reduce token count
    # Screenshots are 5120x2422 (Retina) which generates ~15k image tokens
    # Resize to max 1280px wide to keep token count manageable
    print(f"\n{'='*60}")
    print(f"  Step 4b: Resizing images")
    print(f"{'='*60}")

    MAX_IMAGE_WIDTH = 1280
    resized_count = 0
    for sample in trl_samples:
        images = sample.get("images", [])
        new_images = []
        for img in images:
            if hasattr(img, "width") and img.width > MAX_IMAGE_WIDTH:
                ratio = MAX_IMAGE_WIDTH / img.width
                new_h = int(img.height * ratio)
                img = img.resize((MAX_IMAGE_WIDTH, new_h))
                resized_count += 1
            new_images.append(img)
        sample["images"] = new_images
    print(f"   Resized {resized_count} images to max {MAX_IMAGE_WIDTH}px wide")

    # Step 5: Train with TRL SFTTrainer
    print(f"\n{'='*60}")
    print(f"  Step 5: Starting training")
    print(f"{'='*60}")

    from datasets import Dataset
    from trl import SFTTrainer, SFTConfig

    dataset = Dataset.from_list(trl_samples)

    # Calculate warmup steps (replaces deprecated warmup_ratio)
    total_steps = (len(trl_samples) * args.epochs) // (args.batch_size * 1)
    warmup_steps = 0

    training_args = SFTConfig(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        learning_rate=args.lr,
        warmup_steps=warmup_steps,
        logging_steps=1,
        save_strategy="epoch",
        bf16=True,
        remove_unused_columns=False,
        max_length=8192,
        dataloader_pin_memory=False,  # MPS doesn't support pin_memory
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=processor,
    )

    print("🚀 Training started...")
    trainer.train()

    # Save the final checkpoint
    final_path = Path(args.output) / "final"
    trainer.save_model(str(final_path))
    processor.save_pretrained(str(final_path))

    print(f"\n{'='*60}")
    print(f"  ✅ Training complete!")
    print(f"  Checkpoint saved to: {final_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
