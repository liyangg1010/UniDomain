# HD (Human Demonstration) Extensions for UniDomain

This directory contains enhanced functionality for processing Human Demonstration data where frames are extracted from narrations with temporal gaps. These extensions maintain compatibility with the original UniDomain API while providing specialized handling for datasets with discontinuous temporal structure.

## Overview

The HD extensions were created to handle datasets where:
- **Frames are far apart temporally**: Unlike continuous video, HD data from narrations may have significant time gaps between frames
- **Type annotations need removal**: PDDL type annotations (`?p - person`) are removed for compatibility
- **Custom LLM configurations are required**: Support for various LLM backends and models

## Key Differences from Original UniDomain

| Aspect | Original UniDomain | HD Extensions |
|--------|-------------------|---------------|
| **Frame Processing** | Incremental frame-by-frame learning | Multi-task batch processing with temporal gaps |
| **Type Handling** | Supports typed PDDL | Removes type annotations for compatibility |
| **LLM Support** | Default OpenAI models | Extended model support + thinking tag removal |
| **Domain Fusion** | Standard merging | Enhanced with conflict resolution |

## Installation

The HD extensions are included when you install UniDomain:

```bash
# Install UniDomain as editable package
pip install -e third_party/UniDomain
```

## Components

### 1. Atomic Domain Pipeline for HD

Generate atomic domains from HD data with temporal gaps.

```python
from unidomain import atomic_domain_pipeline_for_hd

success = atomic_domain_pipeline_for_hd(
    data_path="data/tasks.json",           # Batch input: task_id -> {path, instruction}
    save_dir="output/atomic_domain",
    vlm_model="gpt-5",                     # VLM for initial domain
    llm_model="gpt-4o-mini",               # LLM for revision
    log_enabled=True                       # Show detailed logs
)
```

**Input Format** (`tasks.json`):
```json
{
    "task_001": {
        "path": "data/frames/task_001",
        "instruction": "Pick up the cup and place it on the table"
    },
    "task_002": {
        "path": "data/frames/task_002",
        "instruction": "Open the drawer and take out the book"
    }
}
```

### 2. Domain Fusion Pipeline for HD

Fuse multiple atomic domains with type removal and conflict resolution.

```python
from unidomain import domain_fusion_pipeline_hd

domain_fusion_pipeline_hd(
    domain_dir="output/atomic_domains",    # Directory with atomic domains
    output_dir="output/fused_domain",
    num_workers=4,                         # Parallel workers
    llm_model="gpt-4o-mini"                # Model for fusion
)
```

### 3. Custom LLM Configuration

Use custom API endpoints and models:

```python
from unidomain import domain_fusion_pipeline_hd
from openai import OpenAI

# Define custom client factory
def my_client_factory(model_name):
    return OpenAI(
        api_key="your-api-key",
        base_url="https://your-api-endpoint.com/v1"
    )

# Run with custom client
domain_fusion_pipeline_hd(
    domain_dir="atomic_domains",
    output_dir="fused_output",
    llm_model="GLM-4.7",                   # Custom model
    custom_client_factory=my_client_factory
)
```

### 4. LLM Utilities

**Remove Thinking Tags**:
```python
from unidomain import exclude_thinking

output = "<think>reasoning...</think>Answer: 42"
cleaned = exclude_thinking(output)  # "Answer: 42"
```

**Extended Model Costs**:
```python
from unidomain import HD_MODEL_COSTS, get_extended_model_costs

# HD-specific models
print(HD_MODEL_COSTS)
# {
#     "Qwen3-VL-235B-A22B-Instruct": (1.0, 1.0),
#     "gpt-5": (1.25, 10.0),
#     ...
# }

# Combined costs (original + HD)
all_costs = get_extended_model_costs()
```

## Module Structure

```
hd_extensions/
├── __init__.py                        # Public API exports
├── llm_utils_hd.py                    # LLM utilities (thinking removal, model costs)
├── initial_domain_hd.py               # Initial domain extraction for HD data
├── atomic_domain_pipeline_hd.py       # Full atomic domain pipeline
├── predicate_merging_hd.py            # Enhanced predicate merging
├── operator_merging_hd.py             # Enhanced operator merging
├── runner_hd.py                       # Domain fusion runner
└── domain_fusion_pipeline_hd.py       # Full fusion pipeline
```

## Example Workflows

### Workflow 1: Generate Atomic Domain from Narrations

```python
from unidomain import atomic_domain_pipeline_for_hd

# Prepare task data
tasks = {
    "P01_01": {
        "path": "data/epic/frames/P01_01",
        "instruction": "Take out the pan from the drawer"
    },
    # ... more tasks
}

# Save tasks to JSON
import json
with open("tasks.json", "w") as f:
    json.dump(tasks, f)

# Generate atomic domain
success = atomic_domain_pipeline_for_hd(
    data_path="tasks.json",
    save_dir="output/P01_atomic",
    vlm_model="gpt-5",
    llm_model="gpt-4o-mini"
)

if success:
    print("Atomic domain generated successfully!")
    # Output: output/P01_atomic/atomic_domain.json
    #         output/P01_atomic/atomic_domain.pddl
```

### Workflow 2: Fuse Multiple Atomic Domains

```python
from unidomain import domain_fusion_pipeline_hd

# Assuming you have multiple atomic domains:
# atomic_domains/
#     P01_01/atomic_domain.json
#     P01_02/atomic_domain.json
#     P01_03/atomic_domain.json

domain_fusion_pipeline_hd(
    domain_dir="atomic_domains",
    output_dir="fused_output",
    num_workers=2,
    llm_model="gpt-4o-mini"
)

# Output:
# fused_output/
#     0/atomic_domain.json (P01_01, types removed)
#     1/atomic_domain.json (P01_02, types removed)
#     2/atomic_domain.json (P01_03, types removed)
#     3/meta_domain.json (0 + 1 merged)
#     4/meta_domain.json (3 + 2 merged, final)
#     mapping_table.json
```

### Workflow 3: Integration with LEAP Pipeline

```python
from pathlib import Path
from unidomain import atomic_domain_pipeline_for_hd, domain_fusion_pipeline_hd
from unidomain.utils.batch_runner import load_batch_inputs

# Load LEAP narration data
data_root = Path("data/dataset/epic_kitchen")
narrations_path = data_root / "hd_metadata/narrations_dict.json"

# Load and convert to task format
import json
with open(narrations_path) as f:
    narrations = json.load(f)

tasks = {}
for video_id, video_data in narrations.items():
    tasks[video_id] = {
        "path": str(data_root / f"frames/{video_id}"),
        "instruction": " ".join([n["narration"] for n in video_data["narrations"]])
    }

# Save batch input
batch_input_path = data_root / "tasks_batch.json"
with open(batch_input_path, "w") as f:
    json.dump(tasks, f)

# Process each video's atomic domain
atomic_output_dir = data_root / "atomic_domains"
atomic_output_dir.mkdir(exist_ok=True)

for video_id in tasks:
    print(f"Processing {video_id}...")
    success = atomic_domain_pipeline_for_hd(
        data_path=batch_input_path,
        save_dir=atomic_output_dir / video_id,
        vlm_model="gpt-5",
        llm_model="gpt-4o-mini",
        log_enabled=False  # Suppress sub-module logs
    )

# Fuse all atomic domains
fusion_output_dir = data_root / "fused_domain"
domain_fusion_pipeline_hd(
    domain_dir=atomic_output_dir,
    output_dir=fusion_output_dir,
    num_workers=4,
    llm_model="gpt-4o-mini"
)
```

## API Reference

### `atomic_domain_pipeline_for_hd`

Generate atomic domain from HD narration data with temporal gaps.

**Parameters:**
- `data_path` (Path): Path to batch inputs JSON
- `save_dir` (Paths): Directory to save outputs
- `vlm_model` (str): VLM model name for initial domain extraction
- `llm_model` (str): LLM model name for revision and verification
- `log_enabled` (bool): Whether to show detailed logs from sub-modules

**Returns:** `bool` - True if successful, False otherwise

---

### `domain_fusion_pipeline_hd`

Execute complete domain fusion pipeline for HD data.

**Parameters:**
- `domain_dir` (Paths): Directory with subdirectories of atomic domains
- `output_dir` (Paths): Directory to store fused domains
- `num_workers` (int): Number of parallel threads (default: 2)
- `llm_model` (str): Model name for fusion (default: "gpt-4o-mini")
- `custom_client_factory` (callable): Optional function to create custom OpenAI client

**Returns:** `None`

---

### `exclude_thinking`

Remove `<think>...</think>` blocks from LLM outputs.

**Parameters:**
- `llm_output` (Union[str, List[str]]): LLM output with potential thinking blocks

**Returns:** Same type as input with thinking blocks removed

---

### `delete_type`

Delete PDDL type annotations from predicates and operators.

**Parameters:**
- `predicates` (Dict[str, str]): Predicate definitions
- `operators` (Dict[str, str]): Operator definitions

**Returns:** `Tuple[Dict[str, str], Dict[str, str]]` - Updated predicates and operators

## Troubleshooting

### Issue: "Module not found: unidomain.hd_extensions"

**Solution:** Reinstall UniDomain in editable mode:
```bash
pip install -e third_party/UniDomain
```

### Issue: Custom client not working

**Solution:** Ensure your client factory returns an OpenAI-compatible client:
```python
from openai import OpenAI

def factory(model):
    return OpenAI(api_key="...", base_url="...")
```

### Issue: Type annotation errors in PDDL

**Solution:** HD extensions automatically remove type annotations. If you need typed PDDL, use the original `domain_fusion_pipeline` instead.

## Comparison with Original Functions

| Function | Original | HD Extension | Key Difference |
|----------|----------|--------------|----------------|
| Atomic Domain | `atomic_domain_pipeline` | `atomic_domain_pipeline_for_hd` | Multi-task batch, temporal gaps |
| Initial Domain | `run_initial_domain_step` | `run_initial_domain_step_for_hd` | Handles task dict input |
| Domain Fusion | `domain_fusion_pipeline` | `domain_fusion_pipeline_hd` | Type removal, custom clients |
| Predicate Merge | `merge_predicates` | `merge_predicates_hd` | Name conflict resolution |
| Operator Merge | `merge_operators` | `merge_operators_hd` | Enhanced filtering |

## Contributing

When adding new HD-specific functionality:
1. Create new files in `hd_extensions/` directory
2. Follow naming convention: `<module>_hd.py`
3. Export public APIs in `__init__.py`
4. Update main `unidomain/__init__.py` lazy imports
5. Add comprehensive docstrings
6. Update this README

## License

Same as UniDomain main project.

## Citation

If you use HD extensions, please cite both UniDomain and LEAP:

```bibtex
@article{unidomain2024,
  title={UniDomain: Pretraining a Unified PDDL Domain from Real-World Demonstrations},
  author={...},
  journal={...},
  year={2024}
}

@article{leap2024,
  title={LEAP: Learning Egocentric Action Plans from Narrations},
  author={...},
  journal={...},
  year={2024}
}
```
