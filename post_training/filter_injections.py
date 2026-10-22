#!/usr/bin/env python3
"""
open-moe-48b Untrusted Content Quarantine & Prompt Injection Sanitizer
======================================================================
Automated security filter for external retrieval data, web crawls, and tool outputs.
Identifies adversarial prompt injection payloads (system override, secret exfiltration,
unauthorized tool execution) and strictly isolates untrusted text within architectural
boundary delimiters: <untrusted_content> ... </untrusted_content>.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Pattern

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("filter_injections")

# High-precision adversarial regex indicators for indirect injection detection
INJECTION_SIGNATURES: list[tuple[str, Pattern]] = [
    (
        "system_override",
        re.compile(
            r"(?i)(ignore\s+(all\s+)?(previous|prior|above)\s+(instructions|directives|prompts)|system\s+override|disregard\s+(all\s+)?rules)",
        ),
    ),
    (
        "prompt_exfiltration",
        re.compile(
            r"(?i)(output|print|reveal|dump|leak|repeat)\s+(the\s+)?(system\s+prompt|initial\s+prompt|developer\s+instructions|hidden\s+rules)",
        ),
    ),
    (
        "role_hijack",
        re.compile(
            r"(?i)(you\s+are\s+now|act\s+as|pretend\s+to\s+be)\s+(DAN|unconstrained|jailbroken|unaligned|root\s+console|godmode)",
        ),
    ),
    (
        "exfiltration_url",
        re.compile(
            r"(?i)(send|post|exfiltrate|forward|curl|wget)\s+.*https?://[a-zA-Z0-9.\-_/]+\?(data|token|key|secret)=",
        ),
    ),
    (
        "destructive_command",
        re.compile(
            r"(?i)(rm\s+-rf\s+/|drop\s+table|delete\s+from|chmod\s+777\s+/|eval\(|exec\()",
        ),
    ),
]


@dataclass
class ThreatDetection:
    has_injection: bool
    categories: list[str]
    matched_patterns: list[str]


def detect_threats(text: str) -> ThreatDetection:
    """Scans text against prompt injection signatures."""
    categories: list[str] = []
    matches: list[str] = []

    for category, pattern in INJECTION_SIGNATURES:
        match = pattern.search(text)
        if match:
            categories.append(category)
            matches.append(match.group(0))

    return ThreatDetection(
        has_injection=len(categories) > 0,
        categories=categories,
        matched_patterns=matches,
    )


def isolate_untrusted_content(text: str, auto_wrap: bool = True) -> str:
    """
    Enforces strict structural isolation:
    Ensures any untrusted external text is encapsulated within <untrusted_content> tags,
    escaping internal delimiters to prevent boundary escapes.
    """
    # Neutralize nested/crafted delimiter escape attempts
    escaped = text.replace("</untrusted_content>", "&lt;/untrusted_content&gt;")
    escaped = escaped.replace("<untrusted_content>", "&lt;untrusted_content&gt;")

    if auto_wrap and not (text.startswith("<untrusted_content>") and text.endswith("</untrusted_content>")):
        return f"<untrusted_content>\n{escaped.strip()}\n</untrusted_content>"
    return escaped


def process_dataset(
    input_path: Path,
    output_path: Path,
    quarantine_only: bool = False,
) -> tuple[int, int]:
    """
    Scans, detects, and isolates prompt injection payloads in JSONL files.
    """
    logger.info("Scanning dataset %s for indirect injection threats...", input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_records = 0
    threats_detected = 0

    with open(input_path, "r", encoding="utf-8") as in_f, open(output_path, "w", encoding="utf-8") as out_f:
        for line_num, line in enumerate(in_f):
            stripped = line.strip()
            if not stripped:
                continue

            try:
                record = json.loads(stripped)
            except json.JSONDecodeError:
                continue

            total_records += 1
            content = record.get("text", "") or record.get("content", "")
            detection = detect_threats(content)

            if detection.has_injection:
                threats_detected += 1
                record["has_injection_threat"] = True
                record["threat_categories"] = detection.categories

                # Sanitize and quarantine within isolation tags
                record["text"] = isolate_untrusted_content(content)
            else:
                record["has_injection_threat"] = False

            if quarantine_only and not detection.has_injection:
                continue

            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info("Scan Complete: %d records processed, %d injection threats isolated.", total_records, threats_detected)
    return total_records, threats_detected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter and isolate indirect prompt injection threats in dataset corpora."
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        required=True,
        help="Input JSONL file containing documents or tool outputs.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("data/sanitized_data.jsonl"),
        help="Target output destination for sanitized records.",
    )
    parser.add_argument(
        "--quarantine-only",
        action="store_true",
        help="Only write records that contained detected threats (for adversarial DPO training).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input_file.exists():
        logger.error("Input file %s does not exist.", args.input_file)
        sys.exit(1)

    process_dataset(
        input_path=args.input_file,
        output_path=args.output_file,
        quarantine_only=args.quarantine_only,
    )


if __name__ == "__main__":
    main()
