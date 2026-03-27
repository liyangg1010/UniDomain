"""
Example usage of UniDomain HD Extensions for LEAP project.

This script demonstrates how to:
1. Generate atomic domains from HD narration data
2. Fuse multiple atomic domains
3. Use custom LLM configurations

Run this after installing UniDomain:
    pip install -e third_party/UniDomain
"""

import json
from pathlib import Path

# Import HD extensions
from unidomain import (
    atomic_domain_pipeline_for_hd,
    domain_fusion_pipeline_hd,
    exclude_thinking,
)


def example_1_atomic_domain_basic():
    """Example 1: Generate atomic domain from a single task batch."""
    print("=" * 80)
    print("Example 1: Basic Atomic Domain Generation")
    print("=" * 80)

    # Prepare sample tasks
    tasks = {
        "task_001": {
            "path": "data/frames/task_001",
            "instruction": "Pick up the red cup and place it on the table"
        },
        "task_002": {
            "path": "data/frames/task_002",
            "instruction": "Open the drawer and take out the blue book"
        }
    }

    # Save to temporary file
    tasks_file = Path("temp_tasks.json")
    with open(tasks_file, "w") as f:
        json.dump(tasks, f, indent=2)

    print(f"\nTasks saved to: {tasks_file}")
    print(f"Number of tasks: {len(tasks)}")

    # Generate atomic domain
    success = atomic_domain_pipeline_for_hd(
        data_path=tasks_file,
        save_dir="output/example1_atomic",
        vlm_model="gpt-5",
        llm_model="gpt-4o-mini",
        log_enabled=True
    )

    if success:
        print("\n✅ Atomic domain generated successfully!")
        print("Output files:")
        print("  - output/example1_atomic/atomic_domain.json")
        print("  - output/example1_atomic/atomic_domain.pddl")
    else:
        print("\n❌ Atomic domain generation failed!")

    # Cleanup
    tasks_file.unlink(missing_ok=True)


def example_2_domain_fusion_basic():
    """Example 2: Fuse multiple atomic domains."""
    print("\n" + "=" * 80)
    print("Example 2: Basic Domain Fusion")
    print("=" * 80)

    # Assuming you have atomic domains in these directories:
    # atomic_domains/
    #     P01_01/atomic_domain.json
    #     P01_02/atomic_domain.json
    #     P01_03/atomic_domain.json

    domain_dir = Path("atomic_domains")

    if not domain_dir.exists():
        print(f"\n⚠️  Directory not found: {domain_dir}")
        print("Please create atomic domains first using example_1_atomic_domain_basic()")
        return

    print(f"\nFusing domains from: {domain_dir}")
    print(f"Number of atomic domains: {len(list(domain_dir.iterdir()))}")

    # Fuse domains
    domain_fusion_pipeline_hd(
        domain_dir=domain_dir,
        output_dir="output/example2_fused",
        num_workers=2,
        llm_model="gpt-4o-mini"
    )

    print("\n✅ Domain fusion completed!")
    print("Output directory: output/example2_fused/")


def example_3_custom_llm_client():
    """Example 3: Use custom LLM client factory."""
    print("\n" + "=" * 80)
    print("Example 3: Custom LLM Configuration")
    print("=" * 80)

    from openai import OpenAI

    # Define custom client factory
    def qz_client_factory(model_name):
        """Create client for QZ LLM gateway."""
        # This is a placeholder - replace with actual API configuration
        QZ_LLM_CONFIG = {
            "GLM-4.7": {
                "api_key": "your-api-key-here",
                "base_url": "https://your-api-endpoint.com/v1"
            }
        }

        if model_name in QZ_LLM_CONFIG:
            return OpenAI(
                api_key=QZ_LLM_CONFIG[model_name]["api_key"],
                base_url=QZ_LLM_CONFIG[model_name]["base_url"]
            )
        else:
            # Fallback to default OpenAI
            return OpenAI()

    print("\n📝 Custom client factory defined")
    print("Supported models: GLM-4.7")

    # Use with domain fusion
    domain_dir = Path("atomic_domains")

    if domain_dir.exists():
        print(f"\nRunning fusion with custom LLM: GLM-4.7")

        domain_fusion_pipeline_hd(
            domain_dir=domain_dir,
            output_dir="output/example3_fused_custom",
            num_workers=2,
            llm_model="GLM-4.7",
            custom_client_factory=qz_client_factory
        )

        print("\n✅ Custom LLM fusion completed!")
    else:
        print(f"\n⚠️  Directory not found: {domain_dir}")
        print("Please create atomic domains first")


def example_4_thinking_removal():
    """Example 4: Remove thinking tags from LLM output."""
    print("\n" + "=" * 80)
    print("Example 4: Thinking Tag Removal")
    print("=" * 80)

    # Sample LLM outputs with thinking tags
    outputs = [
        "<think>Let me analyze this...</think>The answer is 42",
        "<THINK>Considering the options...</THINK>Option A is correct",
        "No thinking tags here, just plain text",
        "<think>\nMulti-line\nthinking\n</think>Final answer: Yes"
    ]

    print("\nProcessing LLM outputs:")
    for i, output in enumerate(outputs, 1):
        print(f"\n{i}. Original:")
        print(f"   {repr(output)}")

        cleaned = exclude_thinking(output)
        print(f"   Cleaned:")
        print(f"   {repr(cleaned)}")


def example_5_leap_integration():
    """Example 5: Full LEAP pipeline integration."""
    print("\n" + "=" * 80)
    print("Example 5: LEAP Pipeline Integration")
    print("=" * 80)

    # This example shows how to integrate with LEAP's EPIC-KITCHENS data
    data_root = Path("data/dataset/epic_kitchen")
    narrations_path = data_root / "hd_metadata/narrations_dict.json"

    if not narrations_path.exists():
        print(f"\n⚠️  Narrations file not found: {narrations_path}")
        print("Please run EPIC-KITCHENS data preparation first")
        return

    print(f"\nLoading narrations from: {narrations_path}")

    # Load narrations
    with open(narrations_path) as f:
        narrations_dict = json.load(f)

    print(f"Number of videos: {len(narrations_dict)}")

    # Convert to task format (limit to 3 videos for demo)
    tasks = {}
    for video_id in list(narrations_dict.keys())[:3]:
        video_data = narrations_dict[video_id]

        # Check if frames exist
        frames_dir = data_root / f"frames/{video_id}"
        if not frames_dir.exists():
            print(f"⚠️  Frames not found for {video_id}, skipping")
            continue

        # Combine all narrations as instruction
        instruction = " ".join([
            n.get("narration", "") for n in video_data.get("narrations", [])
        ])

        tasks[video_id] = {
            "path": str(frames_dir),
            "instruction": instruction
        }

    print(f"Prepared {len(tasks)} tasks for processing")

    if not tasks:
        print("No valid tasks found")
        return

    # Save batch input
    batch_input_path = data_root / "tasks_batch_demo.json"
    with open(batch_input_path, "w") as f:
        json.dump(tasks, f, indent=2)

    print(f"\nBatch input saved to: {batch_input_path}")

    # Process atomic domains (one per video)
    atomic_output_dir = data_root / "atomic_domains_demo"
    atomic_output_dir.mkdir(exist_ok=True)

    print("\n📝 Generating atomic domains...")
    for video_id in tasks:
        print(f"  Processing {video_id}...")

        success = atomic_domain_pipeline_for_hd(
            data_path=batch_input_path,
            save_dir=atomic_output_dir / video_id,
            vlm_model="gpt-5",
            llm_model="gpt-4o-mini",
            log_enabled=False
        )

        if success:
            print(f"    ✅ Success")
        else:
            print(f"    ❌ Failed")

    # Fuse all atomic domains
    print("\n📝 Fusing atomic domains...")
    fusion_output_dir = data_root / "fused_domain_demo"

    domain_fusion_pipeline_hd(
        domain_dir=atomic_output_dir,
        output_dir=fusion_output_dir,
        num_workers=2,
        llm_model="gpt-4o-mini"
    )

    print("\n✅ LEAP pipeline integration completed!")
    print(f"Atomic domains: {atomic_output_dir}")
    print(f"Fused domain: {fusion_output_dir}")


def main():
    """Run all examples."""
    print("UniDomain HD Extensions - Usage Examples")
    print("=" * 80)

    examples = [
        ("Basic Atomic Domain Generation", example_1_atomic_domain_basic),
        ("Basic Domain Fusion", example_2_domain_fusion_basic),
        ("Custom LLM Configuration", example_3_custom_llm_client),
        ("Thinking Tag Removal", example_4_thinking_removal),
        ("LEAP Pipeline Integration", example_5_leap_integration),
    ]

    print("\nAvailable examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")

    print("\nSelect example to run (1-5, or 'all' for all examples):")
    choice = input("> ").strip().lower()

    if choice == "all":
        for _, func in examples:
            func()
    elif choice.isdigit() and 1 <= int(choice) <= len(examples):
        examples[int(choice) - 1][1]()
    else:
        print("Invalid choice!")


if __name__ == "__main__":
    # Run example 4 by default (no external dependencies)
    print("Running Example 4: Thinking Tag Removal")
    print("(This example has no external dependencies)")
    example_4_thinking_removal()

    print("\n" + "=" * 80)
    print("\nTo run other examples, uncomment the desired function call.")
    print("To run all examples interactively, uncomment: main()")
