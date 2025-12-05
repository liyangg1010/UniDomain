"""
Logging Configuration & Formatters
==================================

This module defines the detailed configuration for the logging system.
It includes custom Formatters for rendering logs in both terminal (colored)
and file (ASCII-boxed) formats, as well as a specialized DispatchingFileHandler
that routes logs to different directories based on runtime context.

Architecture Position:
    [Configuration / Infrastructure] -> Used by utils.logger to initialize logging.
"""

import logging
import shutil
import threading
import time
from logging import LogRecord
from pathlib import Path

from tabulate import tabulate
from termcolor import colored
from wcwidth import wcswidth

from unidomain.configs.constants import File, LoggerName


# =================================================================
# ### Formatter 1: Terminal Color Formatter ###
# =================================================================
class ColorContentFormatter(logging.Formatter):
    """Custom formatter for terminal output using ANSI colors.

    It distinguishes between "llm_agent" logs (showing prompts/responses)
    and generic "system" logs.
    """

    def format(self, record: LogRecord) -> str:
        log_type = getattr(record, "log_type", "system")
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(record.created))

        if log_type == "llm_agent":
            header = colored(f"---LLMAgent @ {timestamp}---", "green", attrs=["bold"])
            call_id_label = colored("call_id:", "yellow")
            prompt_label = colored("---input prompt---", "cyan")
            output_label = colored("---output---", "cyan")
            
            call_id = getattr(record, "call_id", "N/A")
            prompt = getattr(record, "prompt", "N/A")
            response = getattr(record, "response", "N/A")

            return (
                f"{header}\n"
                f"{call_id_label} {call_id}\n"
                f"{prompt_label}\n"
                f"{prompt}\n"
                f"{output_label}\n"
                f"{response}\n"
            )
        
        elif log_type == "system":
            header = colored(f"---System @ {timestamp}---", "blue")
            message = record.getMessage()
            return f"{header}\n{message}\n"
            
        else:
            return super().format(record)
    

# =================================================================
# ### Formatter 2: ASCII Box File Formatter ###
# =================================================================
class FileContentFormatter(logging.Formatter):
    """Custom formatter for file output that wraps content in an ASCII box.

    It ensures proper text wrapping (using wcwidth for CJK support) and
    visual separation between log entries.
    """

    def __init__(self, *args, **kwargs):
        
        super().__init__(*args, **kwargs)
        MIN_WIDTH = 78
        self.width = MIN_WIDTH

        # Attempt to adapt to terminal width if available
        try:
            term_width = shutil.get_terminal_size().columns
            # Reserve some margin
            self.width = max(self.width, term_width)
        except OSError:
            pass    # Fallback to default if not a terminal
        
    def _pad_text(self, text: str, width: int) -> str:
        """"Pad text with spaces to match visual width (handling wide chars)."""
        current_width = wcswidth(text)
        padding = " " * (width - current_width)
        return text + padding

    def _wrap_and_pad_text(self, text: str, width: int) -> str:
        """Wrap text into lines that fit within the box borders."""
        lines = []
        inner_width = width - 2 # Subtract borders '║ ' and ' ║' effectively
        for para in str(text).splitlines():
            if not para:
                lines.append(f"║ {self._pad_text('', inner_width)} ║")
                continue
            
            current_line = ""
            current_width = 0

            for word in para.split():
                word_width = wcswidth(word)

                # Check overflow (+1 for space)
                if current_width + word_width + 1 > inner_width:
                    lines.append(f"║ {self._pad_text(current_line, inner_width)} ║")
                    current_line = word
                    current_width = word_width
                else:
                    if current_line:
                        current_line += " "
                        current_width += 1
                    current_line += word
                    current_width += word_width

            lines.append(f"║ {self._pad_text(current_line, inner_width)} ║")
        return "\n".join(lines)

    def _create_box(self, title: str, timestamp: str, headers: list, content_blocks: dict) -> str:
        """Assemble the ASCII box structure."""
        box = []
        
        # Top Border
        box.append(f"╔{'═' * self.width}╗")
        
        # Header Section
        main_header = self._pad_text(f"{title} @ {timestamp}", self.width - 2)
        box.append(f"║ {main_header} ║")
        for h in headers:
            box.append(f"║ {self._pad_text(h, self.width - 2)} ║")

        # Content Blocks
        for subtitle, text in content_blocks.items():
            box.append(f"╠{'═' * self.width}╣")
            
            # Centered Subtitle
            centered_subtitle_text = f" --- {subtitle} --- "
            padding_total = self.width - 2 - wcswidth(centered_subtitle_text)
            padding_left = padding_total // 2
            padding_right = padding_total - padding_left
            centered_subtitle = f'{" " * padding_left}{centered_subtitle_text}{" " * padding_right}'
            box.append(f"║ {centered_subtitle} ║")
            
            box.append(f"╠{'─' * self.width}╣")
            box.append(self._wrap_and_pad_text(text, self.width))

        # Bottom Border
        box.append(f"╚{'═' * self.width}╝")
        return "\n".join(box) + "\n"

    def format(self, record: LogRecord) -> str:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(record.created))
        log_type = getattr(record, "log_type", "system")

        if log_type == "llm_agent":
            call_id = getattr(record, "call_id", "N/A")
            prompt = getattr(record, "prompt", "N/A")
            response = getattr(record, "response", "N/A")
            
            return self._create_box(
                title="LLMAgent",
                timestamp=timestamp,
                headers=[f"call_id: {call_id}"],
                content_blocks={
                    "input prompt": prompt,
                    "output": response
                }
            )
        
        elif log_type == "system":
            message = record.getMessage()
            
            return self._create_box(
                title="System",
                timestamp=timestamp,
                headers=[],
                content_blocks={
                    "message": message
                }
            )

        else:
            return super().format(record)


# =================================================================
# ### Formatter 3: LLM Usage Table Formatter ###
# =================================================================
class LLMUsageFormatter(logging.Formatter):
    """Custom formatter for LLM usage logs using a PSQL-style table format."""

    def format(self, record: LogRecord) -> str:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(record.created))
        
        rows = [
            ["call_id", getattr(record, "call_id", "N/A")],
            ["model_name", getattr(record, "model_name", "N/A")],
            ["input_tokens", getattr(record, "input_tokens", "N/A")],
            ["output_tokens", getattr(record, "output_tokens", "N/A")],
            ["thinking_time_seconds", getattr(record, "thinking_time_seconds", "N/A")],
            ["cost", getattr(record, "cost", "N/A")],
            ["total_costs_to_now", getattr(record, "total_costs_to_now", "N/A")]
        ]
        
        # Generate table with dummy headers to steal borders
        dummy_headers = ["", ""]
        table_str = tabulate(rows, headers=dummy_headers, tablefmt="psql")
        
        table_lines = table_str.splitlines()
        
        # Extract parts
        top_border = table_lines[0]
        header_separator = table_lines[2]  # The line |---+---|
        table_body = "\n".join(table_lines[3:])  # Data rows
        
        # Create custom title header
        inner_width = len(top_border) - 2
        title_text = f" Metric Record @ {timestamp} "
        centered_title = title_text.center(inner_width)
        custom_header_line = f"|{centered_title}|"
        
        # Assemble
        return (
            f"{top_border}\n"
            f"{custom_header_line}\n"
            f"{header_separator}\n"
            f"{table_body}\n"
        )


# =================================================================
# ### Custom Handler: Context-Aware Dispatcher ###
# =================================================================
class DispatchingFileHandler(logging.Handler):
    """Custom handler that dynamically routes logs to different files based on context.

    It inspects the `log_dir` attribute in the LogRecord (injected by LoggerAdapter)
    and creates/retrieves a FileHandler for that specific directory on the fly.
    """

    def __init__(self, filename: str, **kwargs):
        super().__init__(**kwargs)
        self.filename = filename
        self._handlers = {}
        self._lock = threading.Lock()

    def _get_handler(self, log_path: Path) -> logging.FileHandler:
        """Get or create a cached FileHandler for the given path."""

        log_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            if log_path not in self._handlers:
                handler = logging.FileHandler(log_path, encoding="utf-8")
                handler.setFormatter(self.formatter)
                self._handlers[log_path] = handler
            return self._handlers[log_path]

    def emit(self, record: logging.LogRecord) -> None:
        """Route the log record to the appropriate file handler."""
        # Retrieve context injected by LoggerAdapter
        log_dir = getattr(record, "log_dir", None)

        if not log_dir:
            print(f"[LogWarning] Record missing 'log_dir': {record.getMessage()}")
            return

        try:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            full_path = log_dir / self.filename

            handler = self._get_handler(full_path)
            handler.emit(record)
        except Exception:
            self.handleError(record)


# =================================================================
# ### Global Logging Configuration Dictionary ###
# =================================================================
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,

    "formatters": {
        "color_formatter": {
            "()": ColorContentFormatter,
        },
        "file_formatter": {
            "()": FileContentFormatter,
        },
        "llm_usage_formatter": {
            "()": LLMUsageFormatter,
        },
    },

    "handlers": {
        # --- Execution Loggers ---
        "execution_dispatch_handler": {
            "()": DispatchingFileHandler,
            "filename": f"{File.EXEC_LOG}",
            "formatter": "file_formatter",
            "level": "INFO",
        },
        "execution_console_handler": {
            "class": "logging.StreamHandler",
            "formatter": "color_formatter",
            "level": "INFO",
        },
        
        # --- LLM Usage Loggers ---
        "llm_usage_dispatch_handler": {
            "()": DispatchingFileHandler,
            "filename": f"{File.USAGE_LOG}",
            "formatter": "llm_usage_formatter",
            "level": "INFO",
        },
    },

    "loggers": {
        LoggerName.EXECUTION: {
            "handlers": ["execution_dispatch_handler", "execution_console_handler"],
            "level": "DEBUG",
            "propagate": False,
        },
        LoggerName.LLM_USAGE: {
            "handlers": ["llm_usage_dispatch_handler"],
            "level": "INFO",
            "propagate": False,
        },
    },
    
    "root": {
        "handlers": [],
        "level": "WARNING",
    },
}
