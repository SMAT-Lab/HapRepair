"""Offline integrity and reporting checks. No network, credentials, or model calls."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

def read(path):
    return json.loads(path.read_text())

def check(ok, message):
    if not ok:
        raise SystemExit('FAIL: ' + message)

manifest = read(HERE / 'publication_manifest.json')
for item in manifest['files']:
    path = ROOT / item['path']
    check(path.is_file(), 'missing ' + item['path'])
    data = path.read_bytes()
    check(len(data) == item['bytes'], 'size ' + item['path'])
    check(hashlib.sha256(data).hexdigest() == item['sha256'], 'hash ' + item['path'])
print('PASS:', len(manifest['files']), 'published source/evidence hashes')

corpus = [json.loads(line) for line in (ROOT / 'revision/knowledge_base/rule_complete_383.jsonl').read_text().splitlines() if line.strip()]
check(len(corpus) == 383, '383 corpus pairs')
check(len({(x['namespace'], x['rule']) for x in corpus}) == 63, '63 rule identities')
print('PASS: corpus 383 pairs / 63 identities')

folder = HERE / 'evidence/full_pair_35/campaign_02_results'
rows = read(folder / 'project_repetition_rows.json')
summary = read(folder / 'summary.json')['all_project_repetitions']
check(len(rows) == 70, '70 comparison pairs')
check(len({(r['repeat'], r['project']) for r in rows}) == 70, 'unique paired records')
for condition, expected in [('skill', 70), ('baseline', 58)]:
    check(sum(bool(r[condition]['acceptable_candidate']) for r in rows) == expected, condition + ' accepted candidates')
    check(summary[condition]['acceptable_candidates'] == expected, condition + ' aggregate accepted candidates')
    for row in rows:
        path = HERE / 'run-manifests/rq4' / str(row['repeat']) / condition / row['project'] / Path(row[condition]['manifest']['path']).name
        check(hashlib.sha256(path.read_bytes()).hexdigest() == row[condition]['manifest']['sha256'], 'canonical paired manifest ' + str(path))
print('PASS: canonical 70 comparison pairs / accepted 70 vs 58')

oracle = read(ROOT / 'revision/independent_oracle/final_evaluation_v1/summary.json')
check(oracle['final_results']['total'] == oracle['final_results']['correct'] == 63, 'controlled 63-case assessment')
for name, relative in oracle['provenance'].items():
    check((ROOT / 'revision/independent_oracle/final_evaluation_v1' / relative).exists(), 'oracle provenance ' + name)
print('PASS: controlled assessment and replacement provenance')
print('All offline publication checks passed. No experiments were rerun.')
