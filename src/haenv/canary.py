"""Supported public interface; implementation lives in haenv.provenance.canary."""
from haenv.provenance.canary import (
    EXTRA_ROW_GLOBS,
    FIELD,
    FROZEN_UNTOUCHED,
    HAENV,
    HAENV_GUID,
    JSONL_ROW_GLOBS,
    LINES,
    MANIFEST_KEY,
    MARKER,
    REBUILD,
    RUNTIME_ROW_FILES,
    SURFACE_EXCLUDE_PREFIXES,
    SURFACE_GLOBS,
    block,
    canary_outside_the_case,
    comment_prefix,
    extra_row_files,
    frozen_files,
    generated_files,
    has_canary,
    jsonl_packs,
    main,
    manifest_block,
    manifest_problems,
    md_doc,
    rebuild,
    row_has_canary,
    stamp_jsonl,
    stamp_line,
    stamp_row,
    strip_jsonl,
    strip_row,
    surface,
    txt_sidecar,
    write_manifest_block,
)

__all__ = ['EXTRA_ROW_GLOBS', 'FIELD', 'FROZEN_UNTOUCHED', 'HAENV', 'HAENV_GUID', 'JSONL_ROW_GLOBS', 'LINES', 'MANIFEST_KEY', 'MARKER', 'REBUILD', 'RUNTIME_ROW_FILES', 'SURFACE_EXCLUDE_PREFIXES', 'SURFACE_GLOBS', 'block', 'canary_outside_the_case', 'comment_prefix', 'extra_row_files', 'frozen_files', 'generated_files', 'has_canary', 'jsonl_packs', 'main', 'manifest_block', 'manifest_problems', 'md_doc', 'rebuild', 'row_has_canary', 'stamp_jsonl', 'stamp_line', 'stamp_row', 'strip_jsonl', 'strip_row', 'surface', 'txt_sidecar', 'write_manifest_block']

if __name__ == "__main__":
    raise SystemExit(main())
