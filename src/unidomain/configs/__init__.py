"""
Configuration Module
====================

This module acts as the centralized registry for all system settings, hyperparameters,
and resource paths. It utilizes Pydantic to ensure type safety, validation, and
easy serialization of configuration objects.

The global access point is the `config` singleton, which aggregates specific
settings from all sub-modules.

Module Overview
---------------
- **settings.py**: The top-level entry point. It defines `UniDomainSettings` which
  composes all other config classes and instantiates the global `config` object.

- **constants.py**: Defines project-wide static constants via namespace classes
  (File, Dir, JSONKey, EnvVar, LoggerName) to eliminate magic strings and ensure
  protocol consistency.

- **unidomain.py**: Defines hyperparameters for the core functional pillars:
  Keyframe Extraction, Atomic Domain Generation, Domain Fusion, and Task Planning.

- **llm.py**: Manages Large Language Model settings, including model pricing (USD),
  instantiation parameters, and global retry/reliability policies.

- **prompt_path.py**: Maps logical task names to the physical file paths of
  prompt templates stored in the `prompt_templates/` directory.

- **logger.py**: Contains detailed logging configuration, including custom
  ASCII-art formatters and context-aware file handlers.

- **baselines.py**: Configuration settings specific to benchmark comparison methods.
"""

from unidomain.configs.constants import Dir, EnvVar, File, JSONKey, LoggerName
from unidomain.configs.logger import LOGGING_CONFIG
from unidomain.configs.settings import config

__all__ = [
    "config",
    "LOGGING_CONFIG",
    "Dir",
    "EnvVar",
    "File",
    "JSONKey",
    "LoggerName"
]
