"""Runtime rendering of catalog-backed prompts and versioned judge policies."""
from __future__ import annotations

from haenv.i18n import current
from haenv.i18n.text import L, render, text_of


def runtime_text(value):
    """Render a deferred runtime message using the language selected from the batch."""
    return render(value, current())


def policy_texts(value):
    """Resolve explicit policy text references; preserve all structural policy fields."""
    if isinstance(value, dict):
        if set(value) == {"i18n"}:
            return text_of(value["i18n"])
        return {k: policy_texts(v) for k, v in value.items()}
    if isinstance(value, list):
        return [policy_texts(v) for v in value]
    return value


def registry_phrase(value):
    """A canonical registry phrase, emitted at runtime rather than generation."""
    return runtime_text(L(value))


def read_batch_language(batch):
    """Read the persisted language; an unlabelled historical batch is Chinese."""
    import json
    from pathlib import Path
    from haenv.i18n import batch_language
    path = Path(batch) / 'batch.json'
    return batch_language(json.loads(path.read_text(encoding='utf-8')) if path.is_file() else {})


def validate_rows_language(rows, lang):
    from haenv.i18n import LanguageError, batch_language
    if any(batch_language(row) != lang for row in rows):
        raise LanguageError('Cannot mix row and batch languages')


def select_batch_language(batches, requested=None):
    """Select one language for a judging run; legacy batch metadata means zh."""
    import json
    from pathlib import Path
    from haenv.i18n import LanguageError, check, set_current
    languages = set()
    for batch in batches:
        lang = read_batch_language(batch)
        languages.add(lang)
        rows = Path(batch) / 'eval.jsonl'
        if rows.is_file():
            validate_rows_language([json.loads(line) for line in rows.read_text(encoding='utf-8').splitlines()
                                    if line.strip()], lang)
    if len(languages) > 1:
        raise LanguageError('A judging run cannot mix batch languages')
    lang = next(iter(languages), 'zh')
    if requested is not None and check(requested) != lang:
        raise LanguageError(f'Batch language is {lang}; cannot change it to {requested}; build a fresh batch')
    set_current(lang)
    return lang


def require_assets(lang):
    """Reject absent catalogs before generating questions or dispatching model calls."""
    from haenv.i18n.text import MissingText, templates, lexicon
    if not templates(lang) or not lexicon(lang):
        raise MissingText('language catalogs (templates and lexicon)', lang)


def select_saved_language(manifest):
    """Restore the language of a sealed judging run before rendering any requests."""
    from haenv.i18n import batch_language, set_current
    lang = batch_language(manifest)
    set_current(lang)
    return lang


def localized_judge_version(version, policy):
    """Keep the native archive identity; other languages own a policy stamp."""
    if current() == 'zh':
        return version
    import hashlib
    import json
    from haenv.i18n.text import asset_files
    lang = current()
    body = json.dumps(policy, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    digest = hashlib.sha256(body.encode())
    for path in asset_files(lang):
        if path.name.startswith('judge'):
            digest.update(str(path.relative_to(path.parent.parent)).encode())
            digest.update(path.read_bytes())
    return f'{version}:{lang}:{digest.hexdigest()[:16]}'
