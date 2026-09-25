"""Cache the aida-ugent/llm-ideology-analysis inputs the JEV value-system audit needs.

Three artefacts land in this directory and every step is skipped when its output
already exists, so the notebook can call `prepare_all()` on every run:

  tags.csv               3,991 persons x 61 Manifesto tags (booleans), key `name-en`
  summaries.csv          Wikipedia summaries per person in six languages
  sample.csv             the tag-aware person sample used by every arm
  tag_counts.csv         realised bearers per tag inside the sample
  responses_sample.csv   Stage 1 texts and published Stage 2 labels in all six
                         languages, restricted to sampled persons. All languages
                         are kept because the PCA is fitted on the paper's 77
                         model-by-language respondents; JEV itself only ever
                         reads the English rows.

`responses/valid.csv` is 1.3 GB, so it is streamed through a CSV reader and
filtered on the fly rather than downloaded whole. The datasets-server /filter
endpoint would be tidier but its DuckDB index reports itself corrupt.
"""
from __future__ import annotations

import csv
import io
import random
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = 'https://huggingface.co/datasets/aida-ugent/llm-ideology-analysis/resolve/main'

SEED = 20240917
TARGET_N = 800
RARE_TAG_FLOOR = 60   # tags with fewer bearers than this are taken in full
GAIN_POWER = 1.25     # tunes the sample back onto the population's tags-per-person

TAGS_CSV = HERE / 'tags.csv'
SUMMARIES_CSV = HERE / 'summaries.csv'
SAMPLE_CSV = HERE / 'sample.csv'
TAG_COUNTS_CSV = HERE / 'tag_counts.csv'
RESPONSES_CSV = HERE / 'responses_sample.csv'

RESPONSE_COLUMNS = ['name-en', 'wikidata_id', 'model', 'language',
                    'stage_1_response', 'stage_2_response', 'extracted', 'score']

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


def _download(remote: str, target: Path) -> Path:
    if target.exists():
        return target
    print(f'downloading {remote} -> {target.name}')
    tmp = target.with_suffix(target.suffix + '.tmp')
    urllib.request.urlretrieve(f'{BASE}/{remote}', tmp)
    tmp.replace(target)
    return target


def load_tags():
    """Return (ordered tag names, {person: set of tags})."""
    _download('people_tagged/tags.csv', TAGS_CSV)
    with TAGS_CSV.open(encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    tags = [c for c in rows[0] if c != 'name-en']
    tagged = {r['name-en']: {t for t in tags if r[t] == 'True'} for r in rows}
    return tags, tagged


def load_summaries():
    _download('people_tagged/summaries.csv', SUMMARIES_CSV)
    with SUMMARIES_CSV.open(encoding='utf-8') as handle:
        return {r['name-en']: r for r in csv.DictReader(handle)}


def build_sample():
    """Tag-aware person sample.

    A flat draw would leave the thinnest tags empty: the rarest has 14 bearers in
    the whole dataset. So every bearer of a rare tag is taken, zero-tag persons
    are kept at their population share to anchor the untagged baseline, and the
    remainder is filled greedily toward under-represented tags.
    """
    if SAMPLE_CSV.exists() and TAG_COUNTS_CSV.exists():
        with SAMPLE_CSV.open(encoding='utf-8') as handle:
            return [r['name-en'] for r in csv.DictReader(handle)]

    tags, tagged = load_tags()
    summaries = load_summaries()
    names = sorted(tagged)
    bearers = {t: [n for n in names if t in tagged[n]] for t in tags}

    selected = set()
    for tag in (t for t in tags if len(bearers[t]) < RARE_TAG_FLOOR):
        selected.update(bearers[tag])

    rng = random.Random(SEED)
    zero_tag = [n for n in names if not tagged[n]]
    selected |= set(rng.sample(zero_tag, min(round(TARGET_N * len(zero_tag) / len(names)),
                                             len(zero_tag))))

    counts = {t: 0 for t in tags}
    for name in selected:
        for tag in tagged[name]:
            counts[tag] += 1

    remaining = [n for n in names if n not in selected and tagged[n]]
    rng.shuffle(remaining)
    while len(selected) < TARGET_N and remaining:
        best, best_gain = None, -1.0
        for name in remaining:
            gain = (sum(1.0 / (1 + counts[t]) for t in tagged[name])
                    / len(tagged[name]) ** GAIN_POWER)
            if gain > best_gain:
                best, best_gain = name, gain
        selected.add(best)
        remaining.remove(best)
        for tag in tagged[best]:
            counts[tag] += 1

    sample = sorted(selected)
    with SAMPLE_CSV.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['name-en', 'wikidata_id', 'n_tags'])
        for name in sample:
            writer.writerow([name, summaries[name]['wikidata_id'], len(tagged[name])])

    realised = {t: sum(1 for n in sample if t in tagged[n]) for t in tags}
    with TAG_COUNTS_CSV.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['tag', 'bearers_in_sample', 'bearers_in_dataset'])
        for tag in tags:
            writer.writerow([tag, realised[tag], len(bearers[tag])])
    return sample


def build_responses(sample=None):
    """Stream the 1.3 GB response file, keeping every row for sampled persons."""
    if RESPONSES_CSV.exists():
        return RESPONSES_CSV
    wanted = set(sample if sample is not None else build_sample())
    print(f'streaming responses/valid.csv for {len(wanted)} persons (1.3 GB, no local copy)')

    kept = scanned = 0
    request = urllib.request.Request(f'{BASE}/responses/valid.csv',
                                     headers={'User-Agent': 'jev-value-audit'})
    tmp = RESPONSES_CSV.with_suffix('.tmp')
    with urllib.request.urlopen(request) as raw, tmp.open('w', newline='', encoding='utf-8') as out:
        reader = csv.DictReader(io.TextIOWrapper(raw, encoding='utf-8', newline=''))
        writer = csv.DictWriter(out, fieldnames=RESPONSE_COLUMNS, extrasaction='ignore')
        writer.writeheader()
        for row in reader:
            scanned += 1
            if row['name-en'] in wanted:
                writer.writerow(row)
                kept += 1
            if scanned % 50000 == 0:
                print(f'  scanned {scanned:,} kept {kept:,}', flush=True)
    tmp.replace(RESPONSES_CSV)
    print(f'scanned {scanned:,} rows, kept {kept:,}')
    return RESPONSES_CSV


def prepare_all():
    load_tags()
    load_summaries()
    sample = build_sample()
    build_responses(sample)
    return sample


if __name__ == '__main__':
    names = prepare_all()
    print(f'sample: {len(names)} persons')
