"""Summarize exact-chunk eligible tasks; retain per-query failures."""
import json
from pathlib import Path
import sys

data = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
rows = data['queries']
def match(doc, target):
    return all(doc.get(a) == target.get(b) for a, b in [('file','file'), ('name','function'), ('line_start','line_start'), ('line_end','line_end')])
def score(r, arm):
    targets = [v['target'] for v in r['coverage']]
    if arm in ('vector', 'bm25'):
        docs = r['retrieval']['raw']['rankings'].get(arm, [])
    else: docs = r['retrieval'][arm]['rrf']
    rank = next((i for i, doc in enumerate(docs, 1) if any(match(doc, t) for t in targets)), None)
    return (int(rank is not None), 1 / rank if rank else 0)
eligible = [r for r in rows if r['coverage'] and all(c['vector'] and c['bm25'] for c in r['coverage']) and not any(r['retrieval'][a]['errors'] for a in ('raw','translation_only'))]
out = {'eligible_tasks': sorted({r['task_id'] for r in eligible}), 'eligible_queries': len(eligible), 'source_index_unchanged': data['source_index_unchanged'], 'groups': {}, 'failures': [], 'graph': {}, 'rewrite_status': {}}
for r in rows:
    status = r['retrieval']['translation_only']['plan']['status']; out['rewrite_status'][status] = out['rewrite_status'].get(status, 0) + 1
for group in ['all','en','fr','zh','mixed']:
    selected = [r for r in eligible if group == 'all' or r['language'] == group]
    out['groups'][group] = {'n': len(selected)}
    for arm in ('vector','bm25','raw','translation_only'):
        vals = [score(r,arm) for r in selected]
        out['groups'][group][arm] = {'hit5': sum(v[0] for v in vals)/len(vals) if vals else None, 'mrr5': sum(v[1] for v in vals)/len(vals) if vals else None}
    for r in selected:
        if group != 'all': continue
        out['failures'].append({'query': r['query_id'], 'hit': {a: score(r,a)[0] for a in ('vector','bm25','raw','translation_only')}})
for task in ('caller','cross'):
    subset = [r for r in eligible if r['task_id'] == task]
    out['graph'][task] = {arm: {'all': sum(r['graph'][arm]['metrics']['all_targets'] for r in subset), 'n': len(subset), 'mean_context_characters': sum(r['graph'][arm]['context_characters'] for r in subset)/len(subset) if subset else None} for arm in ('off','qualified','grounded')}
print(json.dumps(out, ensure_ascii=False, indent=2))
