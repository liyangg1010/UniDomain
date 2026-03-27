# UniDomain HD Extensions - Implementation Summary

## Overview

Based on modifications from the `ahat/third_party/unidomain-private` repository, this implementation adds HD (Human Demonstration) extensions to UniDomain for processing egocentric video data with temporal gaps.

## Implementation Approach

**Strategy**: Create new files in a separate `hd_extensions/` directory rather than modifying original UniDomain files.

**Rationale**:
- Maintains original UniDomain functionality intact
- Easy to update/remove HD extensions
- Clear separation of concerns
- Minimal risk of breaking existing code

## Files Created

### Core Modules

1. **`hd_extensions/__init__.py`**
   - Exports all HD extension functions
   - Version info and documentation

2. **`hd_extensions/llm_utils_hd.py`**
   - `exclude_thinking()`: Removes `<think>...</think>` tags from LLM outputs
   - `HD_MODEL_COSTS`: Extended model cost configurations
   - `get_extended_model_costs()`: Combines default and HD model costs

3. **`hd_extensions/initial_domain_hd.py`**
   - `initiate_domain_from_keyframes_for_hd()`: Multi-task batch processing
   - `run_initial_domain_step_for_hd()`: Wrapper for initial domain step
   - Handles temporal gaps in frame sequences

4. **`hd_extensions/atomic_domain_pipeline_hd.py`**
   - `atomic_domain_pipeline_for_hd()`: Full pipeline for HD data
   - `_suppress_submodule_logs()`: Context manager for log control
   - `_update_current_domain()`: Domain file management
   - Adapted for frame sequences with temporal discontinuity

5. **`hd_extensions/predicate_merging_hd.py`**
   - `merge_predicates_hd()`: Enhanced predicate merging
   - Name conflict resolution with hash suffixes
   - Example: `on(?x, ?y)` conflicts → `on_a3f(?x, ?y)`

6. **`hd_extensions/operator_merging_hd.py`**
   - `merge_operators_hd()`: Enhanced operator merging
   - Improved semantic similarity filtering

7. **`hd_extensions/runner_hd.py`**
   - `run_domain_fusion_hd()`: Multi-threaded fusion runner
   - `fuse_two_domains_hd()`: Pairwise domain fusion
   - `_domain_fusion_node_wrapper_hd()`: Thread-safe node processing
   - Support for custom LLM client factories

8. **`hd_extensions/domain_fusion_pipeline_hd.py`**
   - `domain_fusion_pipeline_hd()`: Full fusion pipeline
   - `delete_type()`: Removes PDDL type annotations
   - `_prepare_domain_fusion_workspace_hd()`: Workspace setup with type removal

### Documentation & Testing

9. **`hd_extensions/README.md`**
   - Comprehensive usage guide
   - API reference
   - Example workflows
   - Comparison tables

10. **`hd_extensions/example_usage.py`**
    - 5 complete usage examples
    - Interactive demo script

11. **`hd_extensions/test_installation.py`**
    - Installation verification
    - Function tests
    - Import tests

### Modified Files

12. **`unidomain/__init__.py`**
    - Added lazy imports for all HD functions
    - Added TYPE_CHECKING imports for IDE support

## Key Features

### 1. Temporal Gap Handling

**Problem**: Original UniDomain assumes continuous frame sequences.

**Solution**: HD extensions process frames in task batches where each task has its own temporal context.

```python
tasks = {
    "task_001": {
        "path": "frames/task_001",
        "instruction": "Pick up cup"
    }
}
atomic_domain_pipeline_for_hd(data_path="tasks.json", ...)
```

### 2. Type Annotation Removal

**Problem**: Some PDDL planners don't support typed parameters.

**Solution**: Automatic type removal: `?x - block` → `?x`

```python
predicates, operators = delete_type(predicates, operators)
```

### 3. Name Conflict Resolution

**Problem**: Two predicates with same name but different semantics fail to merge.

**Solution**: Add hash suffix to conflicting predicate:

```
Domain 1: on(?x, ?y) - "x is physically on y"
Domain 2: on(?x, ?y) - "x is attached to y"
Result:   on(?x, ?y) and on_a3f(?x, ?y)
```

### 4. Custom LLM Support

**Problem**: Need to use various LLM backends (not just OpenAI).

**Solution**: Client factory pattern:

```python
def my_factory(model):
    return OpenAI(api_key="...", base_url="...")

domain_fusion_pipeline_hd(
    ...,
    llm_model="GLM-4.7",
    custom_client_factory=my_factory
)
```

### 5. Thinking Tag Removal

**Problem**: Some models (Qwen3-thinking) output reasoning in `<think>` tags.

**Solution**: Post-processing to remove tags:

```python
output = "<think>...</think>Answer: 42"
cleaned = exclude_thinking(output)  # "Answer: 42"
```

## Comparison with Original

| Feature | Original | HD Extensions |
|---------|----------|---------------|
| **Frame Processing** | Sequential, continuous | Batch, with temporal gaps |
| **Input Format** | Single keyframes dir | Dict of task paths + instructions |
| **Type Handling** | Typed PDDL | Untyped PDDL (types removed) |
| **Predicate Merge** | Standard | With conflict resolution |
| **LLM Backend** | OpenAI default | Pluggable client factory |
| **Model Support** | Standard models | Extended models + thinking removal |
| **Log Control** | Fixed | Configurable suppression |

## Import Paths

All HD functions are available via lazy imports from main `unidomain` module:

```python
from unidomain import (
    # Atomic Domain
    atomic_domain_pipeline_for_hd,
    run_initial_domain_step_for_hd,

    # Domain Fusion
    domain_fusion_pipeline_hd,
    delete_type,
    merge_predicates_hd,
    merge_operators_hd,

    # Utilities
    exclude_thinking,
    HD_MODEL_COSTS,
    get_extended_model_costs,
)
```

## Installation & Testing

### Install
```bash
pip install -e third_party/UniDomain
```

### Test
```bash
python third_party/UniDomain/src/unidomain/hd_extensions/test_installation.py
```

### Expected Output
```
🎉 All tests passed! HD extensions are ready to use.
```

## Usage Example

```python
from unidomain import (
    atomic_domain_pipeline_for_hd,
    domain_fusion_pipeline_hd
)

# Step 1: Generate atomic domain from HD data
success = atomic_domain_pipeline_for_hd(
    data_path="epic_tasks.json",
    save_dir="output/atomic",
    vlm_model="gpt-5",
    llm_model="gpt-4o"
)

# Step 2: Fuse multiple atomic domains
domain_fusion_pipeline_hd(
    domain_dir="output/atomic_domains",
    output_dir="output/fused",
    num_workers=4
)
```

## File Structure

```
third_party/UniDomain/src/unidomain/
├── __init__.py                          # Modified: added HD imports
├── hd_extensions/                       # New directory
│   ├── __init__.py                      # Public API
│   ├── llm_utils_hd.py                  # LLM utilities
│   ├── initial_domain_hd.py             # Initial domain for HD
│   ├── atomic_domain_pipeline_hd.py     # Atomic domain pipeline
│   ├── predicate_merging_hd.py          # Predicate merging
│   ├── operator_merging_hd.py           # Operator merging
│   ├── runner_hd.py                     # Fusion runner
│   ├── domain_fusion_pipeline_hd.py     # Fusion pipeline
│   ├── README.md                        # Documentation
│   ├── example_usage.py                 # Examples
│   └── test_installation.py             # Tests
└── ... (original UniDomain files unchanged)
```

## Integration with LEAP

The HD extensions are designed for LEAP's EPIC-KITCHENS pipeline:

```python
# Load EPIC-KITCHENS narrations
narrations = load_narrations("hd_metadata/narrations_dict.json")

# Convert to task format
tasks = {
    video_id: {
        "path": f"frames/{video_id}",
        "instruction": " ".join(narrations[video_id])
    }
    for video_id in narrations
}

# Generate atomic domains
for video_id, task_info in tasks.items():
    atomic_domain_pipeline_for_hd(...)

# Fuse all domains
domain_fusion_pipeline_hd(...)
```

## Dependencies

Same as original UniDomain:
- openai
- pydantic
- networkx
- matplotlib
- etc.

No additional dependencies required.

## Testing Status

✅ All imports successful
✅ exclude_thinking() works correctly
✅ delete_type() removes type annotations
✅ HD_MODEL_COSTS includes extended models
✅ get_extended_model_costs() combines models

## Future Enhancements

Potential improvements:
1. Add more LLM backend examples (Azure, Anthropic, etc.)
2. Add batch processing for domain fusion
3. Add progress tracking for long-running operations
4. Add caching for LLM responses
5. Add metrics/telemetry for HD-specific operations

## Notes

1. **Backward Compatibility**: Original UniDomain functions unchanged
2. **Naming Convention**: All HD functions end with `_hd` suffix
3. **Import Style**: Lazy imports for performance
4. **Code Quality**: Comprehensive docstrings and type hints
5. **Testing**: Installation test covers all public APIs

## Related Files in LEAP

- `scripts/epic-kitchens/extract_nodes_offline.py`: Uses frame extraction
- `src/leap/pipeline/config.py`: Pipeline configuration
- `data/dataset/epic_kitchen/hd_metadata/`: Narration data

## Contact

For questions or issues with HD extensions:
- Check `hd_extensions/README.md` for detailed docs
- Run `hd_extensions/test_installation.py` to verify setup
- See `hd_extensions/example_usage.py` for usage examples
