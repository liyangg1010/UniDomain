"""
UniDomain: Pretraining a Unified PDDL Domain from Real-World
Demonstrations for Generalizable Robot Task Planning
==================================================

UniDomain is a framework that pre-trains executable PDDL domains from large-scale
robot manipulation videos for online task planning.

It extracts atomic domains from visual demonstrations and systematically fuses
them into high-quality meta-domains. This approach supports compositional
generalization, enabling the zero-shot solution of complex, long-horizon
tasks with superior success rates and plan optimality.

Main Components
---------------
1. **Keyframe Extraction**: Compresses video into significant visual states.
2. **Atomic Domain**: Generates executable PDDL atomic domains from visual observations via VLM+LLM.
3. **Domain Fusion**: Merges multiple atomic domains into a meta-domain knowledge base.
4. **Task Planner**: Solves downstream tasks using the fused meta-domain.
"""

__version__ = "1.0.0"

import importlib
from typing import TYPE_CHECKING

_LAZY_IMPORTS = {
    # Config
    "config": "unidomain.configs.settings",
    
    # Keyframes
    "keyframes_pipeline": "unidomain.keyframes.pipeline",
    
    # Atomic Domain
    "atomic_domain_pipeline": "unidomain.atomic_domain.pipeline",
    "atomic_domain_batch_pipeline": "unidomain.atomic_domain.pipeline",
    
    # Domain Fusion
    "domain_fusion_pipeline": "unidomain.domain_fusion.pipeline",
    
    # Task Planner
    "task_planner_pipeline": "unidomain.task_planner.pipeline",
    "task_planner_batch_pipeline": "unidomain.task_planner.pipeline",
}

# Provide static type hints for IDEs
if TYPE_CHECKING:
    from unidomain.atomic_domain.pipeline import atomic_domain_batch_pipeline, atomic_domain_pipeline
    from unidomain.configs.settings import config
    from unidomain.domain_fusion.pipeline import domain_fusion_pipeline
    from unidomain.keyframes.pipeline import keyframes_pipeline
    from unidomain.task_planner.pipeline import task_planner_batch_pipeline, task_planner_pipeline


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_path = _LAZY_IMPORTS[name]
        module = importlib.import_module(module_path)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

def __dir__():
    return list(globals().keys()) + list(_LAZY_IMPORTS.keys())

__all__ = list(_LAZY_IMPORTS.keys()) + ["__version__"]
