#!/usr/bin/env python3
"""Run structural and provenance checks on the packaged repository."""
from pathlib import Path
from collections import Counter, defaultdict
import ast, hashlib, json, sys
ROOT=Path(__file__).resolve().parents[1]

def check(condition, message):
    if not condition: raise AssertionError(message)
    print(f"PASS  {message}")

def read_json(path): return json.loads(path.read_text(encoding="utf-8"))
def read_jsonl(path): return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

def main():
    source=read_jsonl(ROOT/'data/social_reasoning/ProlificAI.jsonl')
    groups=defaultdict(list)
    for row in source: groups[str(row['question']).strip()].append(row)
    sizes=Counter(len(v) for v in groups.values())
    check(len(source)==3820,'ProlificAI source has 3,820 records')
    check(len(groups)==955 and sizes==Counter({4:955}),'source contains 955 exact question groups with four records each')
    copeland=read_jsonl(ROOT/'data/social_reasoning/reference_responses/most_preferred.jsonl')
    check(len(copeland)==955 and len({x['question'] for x in copeland})==955,'Copeland reference set has 955 distinct questions')
    bad=[]
    for x in copeland:
        q=x['question']; ans=' '.join(str(x['answer']).split())
        candidates={' '.join(str(r[k]).split()) for r in groups[q] for k in ('chosen','rejected')}
        if ans not in candidates: bad.append(q)
    check(not bad,'every Copeland response occurs as a source chosen/rejected candidate')

    primary=read_json(ROOT/'results/social_reasoning/primary_clustered_307/social_reasoning_results.json')
    sensitivity=read_json(ROOT/'results/social_reasoning/copeland_single_reference_393/social_reasoning_results.json')
    models=['Qwen3-8B','Qwen3-8B-Base','llama-3.1-8b-base','llama-3.1-8B-instruct']
    pids=primary['model_results'][models[0]]['question_ids']; uids=sensitivity['model_results'][models[0]]['question_ids']
    check(len(pids)==307 and all(primary['model_results'][m]['question_ids']==pids for m in models),'primary sample aligns on the same 307 question groups across all checkpoints')
    check(len(uids)==393 and all(sensitivity['model_results'][m]['question_ids']==uids for m in models),'Copeland sample aligns on the same 393 questions across all checkpoints')
    for m in models:
        r=primary['model_results'][m]
        check(len(r['losses'])==1228 and len(r['question_scores'])==307,'primary records/scores shape for '+m)
        check(abs(sum(r['question_scores'])/307-r['mean_loss'])<1e-10,'primary saved mean matches saved question scores for '+m)
        u=sensitivity['model_results'][m]
        check(len(u['losses'])==393 and abs(sum(u['losses'])/393-u['mean_loss'])<1e-10,'Copeland saved mean matches item losses for '+m)
    check(len(set(pids)&{u[:16] for u in uids})==124,'primary and Copeland samples share 124 question IDs')

    refusal=read_json(ROOT/'data/behavioral_assessment/refusal/refusal_eval_dataset.json')
    check(sum(len(v['prompts']) for v in refusal['harmful'].values())==1280 and len(refusal['harmless'])==320,'refusal prompt corpus has 1,280 harm-adjacent and 320 harmless prompts')
    ai=read_json(ROOT/'data/behavioral_assessment/ai_self_identification_assistive_behavior/ai_awareness_eval_dataset.json')
    check(len(ai['prompts'])==630,'AI self-identification and assistive-behavior prompt corpus has 630 prompts')
    for path in ROOT.rglob('*.py'):
        compile(path.read_text(encoding='utf-8'),str(path),'exec')
    check(True,'all Python scripts parse successfully')

    notebooks=sorted((ROOT/'notebooks').rglob('*.ipynb'))
    check(len(notebooks)==3,'all three Social Reasoning notebooks are present')
    for nb_path in notebooks:
        nb=json.loads(nb_path.read_text(encoding='utf-8'))
        check(isinstance(nb.get('cells'),list),f'notebook structure is valid: {nb_path.relative_to(ROOT)}')
        for i,cell in enumerate(nb['cells']):
            if cell.get('cell_type')=='code':
                try:
                    ast.parse(''.join(cell.get('source',[])))
                except SyntaxError as exc:
                    raise AssertionError(f'invalid code cell {i} in {nb_path.relative_to(ROOT)}: {exc}')
    check(True,'all notebook code cells parse as Python')
    print('\nRepository validation completed successfully.')

if __name__=='__main__':
    try: main()
    except Exception as exc:
        print(f'VALIDATION FAILED: {exc}',file=sys.stderr); raise
