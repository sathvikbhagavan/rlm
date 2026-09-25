# Reviewer source-code evidence

This directory preserves source supplied voluntarily with human-baseline exports.
Each subdirectory is keyed by anonymous reviewer ID and includes a source manifest
with the original archive checksum. Dataset copies, pickle caches, bytecode, and
other binary state are deliberately excluded. The files are evidence of the
reviewer's method, not canonical benchmark implementations; known discrepancies
are documented in `TIER3_TIER4_AUDIT.md`.

Use `human_eval/tools/archive_reviewer_code.py` to reproduce the sanitized copy.
