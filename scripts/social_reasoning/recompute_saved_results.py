#!/usr/bin/env python3
"""Recompute and audit the saved primary and sensitivity Social Reasoning analyses.

This script does not load model weights and does not rerun inference. It
recomputes model means, paired t-tests, effect sizes, percentile-bootstrap CIs,
Holm-adjusted p-values, and the post hoc matched-overlap summaries from the
per-record/per-question losses saved in the result JSONs. It checks the
recomputed primary and sensitivity statistics against the JSON fields stored
with the notebooks, then exports CSVs and an audit report.

Dependencies: numpy, scipy.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data/social_reasoning/ProlificAI.jsonl"
COPELAND = ROOT / "data/social_reasoning/reference_responses/most_preferred.jsonl"
PRIMARY = ROOT / "results/social_reasoning/primary_clustered_307/social_reasoning_results.json"
SENSITIVITY = ROOT / "results/social_reasoning/copeland_single_reference_393/social_reasoning_results.json"
OUTPUT_DIR = ROOT / "results/social_reasoning/audit"
MODELS = ["Qwen3-8B", "Qwen3-8B-Base", "llama-3.1-8b-base", "llama-3.1-8B-instruct"]
LABELS = {
    "Qwen3-8B": "Qwen3 (8B) Instruct",
    "Qwen3-8B-Base": "Qwen3 (8B) Base",
    "llama-3.1-8b-base": "Llama 3.1 (8B) Base",
    "llama-3.1-8B-instruct": "Llama 3.1 (8B) Instruct",
}
N_BOOT_PRIMARY = 5000
SEED_PRIMARY = 42
N_BOOT_COPELAND = 10000
SEED_COPELAND = 2026


def load_jsonl(path):
    rows=[]
    with path.open(encoding="utf-8") as f:
        for n,line in enumerate(f,1):
            if line.strip():
                row=json.loads(line); row["_line"] = n; rows.append(row)
    return rows


def norm_answer(s):
    return " ".join(str(s or "").replace("\r\n","\n").replace("\r","\n").split())


def bootstrap_ci_loop(values, replicates=5000, seed=42):
    """Per-resample loop, matching the primary notebook's RNG call sequence."""
    values=np.asarray(values,dtype=float); rng=np.random.default_rng(seed)
    boot=np.empty(replicates,dtype=float); n=len(values)
    for b in range(replicates):
        boot[b]=np.mean(values[rng.integers(0,n,size=n)])
    lo,hi=np.percentile(boot,[2.5,97.5])
    return float(lo),float(hi)


def bootstrap_ci_batched(values, replicates=10000, seed=2026):
    """Batched percentile CI matching the Copeland notebook's computation."""
    values=np.asarray(values,dtype=float); rng=np.random.default_rng(seed)
    boot=np.empty(replicates,dtype=float); batch=500; n=len(values)
    for start in range(0,replicates,batch):
        take=min(batch,replicates-start)
        idx=rng.integers(0,n,size=(take,n))
        boot[start:start+take]=values[idx].mean(axis=1)
    lo,hi=np.percentile(boot,[2.5,97.5])
    return float(lo),float(hi)


def holm(pvals):
    p=np.asarray(pvals,dtype=float); order=np.argsort(p); sorted_adjusted=np.empty_like(p); running=0.0
    for rank,idx in enumerate(order):
        running=max(running,min(1.0,(len(p)-rank)*p[idx]))
        sorted_adjusted[rank]=running
    result=np.empty_like(p)
    for rank,idx in enumerate(order): result[idx]=sorted_adjusted[rank]
    return result.tolist()


def write_csv(path, rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:
        path.write_text("",encoding="utf-8"); return
    with path.open("w",newline="",encoding="utf-8") as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)


def verify_close(actual, expected, label, atol=1e-10):
    if not math.isclose(float(actual),float(expected),rel_tol=0.0,abs_tol=atol):
        raise ValueError(f"Saved result mismatch for {label}: recomputed={actual!r}, stored={expected!r}")


def load_primary_maps(d):
    maps={}
    for m in MODELS:
        r=d["model_results"][m]
        if len(r["question_ids"])!=len(r["question_scores"]): raise ValueError(f"Primary IDs/scores mismatch for {m}")
        if len(set(r["question_ids"]))!=len(r["question_ids"]): raise ValueError(f"Duplicate primary question IDs for {m}")
        maps[m]=dict(zip(r["question_ids"],map(float,r["question_scores"])))
    return maps


def load_sensitivity_maps(d):
    maps={}
    for m in MODELS:
        r=d["model_results"][m]
        if len(r["question_ids"])!=len(r["losses"]): raise ValueError(f"Copeland IDs/losses mismatch for {m}")
        if len(set(r["question_ids"]))!=len(r["question_ids"]): raise ValueError(f"Duplicate Copeland question IDs for {m}")
        maps[m]=dict(zip(r["question_ids"],map(float,r["losses"])))
    return maps


def paired_bootstrap_stats(x,y, *, seed, replicates, batching):
    x=np.asarray(x,dtype=float); y=np.asarray(y,dtype=float); diff=x-y
    t,p=stats.ttest_rel(x,y,nan_policy="raise")
    dz=float(diff.mean()/diff.std(ddof=1)) if diff.std(ddof=1)>0 else 0.0
    if batching:
        lo,hi=bootstrap_ci_batched(diff,replicates=replicates,seed=seed)
    else:
        lo,hi=bootstrap_ci_loop(diff,replicates=replicates,seed=seed)
    return {"mean_loss_diff":float(diff.mean()),"ci95_lower":lo,"ci95_upper":hi,"paired_dz":dz,"t_statistic":float(t),"p_raw":float(p)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-overlap",type=int,default=10000,help="Bootstrap replicates for the post hoc overlap analysis")
    args=parser.parse_args()
    OUTPUT_DIR.mkdir(parents=True,exist_ok=True)

    # Validate the raw source structure.
    source_rows=load_jsonl(SOURCE); groups=defaultdict(list)
    for row in source_rows:
        q=str(row.get("question","")).strip()
        if not q or not row.get("chosen") or not row.get("rejected"):
            raise ValueError(f"Missing question/chosen/rejected at source line {row['_line']}")
        groups[q].append(row)
    group_sizes=Counter(len(v) for v in groups.values())
    if len(source_rows)!=3820 or len(groups)!=955 or group_sizes!=Counter({4:955}):
        raise ValueError(f"Unexpected source structure: rows={len(source_rows)}, questions={len(groups)}, sizes={group_sizes}")

    # Validate the reduced reference file against pairwise candidates.
    reduced=load_jsonl(COPELAND); reduced_map={}
    for row in reduced:
        q=str(row.get("question","")).strip(); ans=norm_answer(row.get("answer",""))
        if q in reduced_map: raise ValueError(f"Duplicate Copeland question: {q[:100]}")
        if q not in groups: raise ValueError(f"Copeland question missing from source: {q[:100]}")
        candidates={norm_answer(r[k]) for r in groups[q] for k in ("chosen","rejected")}
        if ans not in candidates: raise ValueError(f"Copeland response not found among source candidates: {q[:100]}")
        reduced_map[q]=ans
    if len(reduced_map)!=955: raise ValueError(f"Expected 955 reduced questions, found {len(reduced_map)}")

    p=json.loads(PRIMARY.read_text(encoding="utf-8")); u=json.loads(SENSITIVITY.read_text(encoding="utf-8"))
    p_maps=load_primary_maps(p); u_maps=load_sensitivity_maps(u)
    p_ids=p["model_results"][MODELS[0]]["question_ids"]; u_ids=u["model_results"][MODELS[0]]["question_ids"]
    if len(p_ids)!=307 or any(p["model_results"][m]["question_ids"]!=p_ids for m in MODELS):
        raise ValueError("Primary results must align over exactly 307 ordered question groups")
    if len(u_ids)!=393 or any(u["model_results"][m]["question_ids"]!=u_ids for m in MODELS):
        raise ValueError("Copeland results must align over exactly 393 ordered questions")

    # Resolve sampled question IDs back to exact source question text and line numbers.
    sha_to_q={hashlib.sha256(q.encode("utf-8")).hexdigest():q for q in groups}
    prefix_to_sha={sha[:16]:sha for sha in sha_to_q}
    manifest=[]
    for sample,ids,is_prefix in [("primary_clustered_307",p_ids,True),("copeland_single_reference_393",u_ids,False)]:
        for qid in ids:
            full_sha=prefix_to_sha.get(qid) if is_prefix else qid
            if not full_sha or full_sha not in sha_to_q: raise ValueError(f"Sample ID does not map to source question: {qid}")
            q=sha_to_q[full_sha]
            manifest.append({"sample":sample,"question_sha256":full_sha,"question_text":q,"source_line_numbers":";".join(str(r["_line"]) for r in groups[q]),"n_source_records":len(groups[q])})
    write_csv(OUTPUT_DIR/"selected_question_manifest.csv",manifest)

    # Recompute primary model means and CIs; verify exact agreement with saved JSON.
    primary_models=[]
    for m in MODELS:
        r=p["model_results"][m]; values=np.asarray(r["question_scores"],dtype=float)
        mean=float(values.mean()); lo,hi=bootstrap_ci_loop(values,N_BOOT_PRIMARY,SEED_PRIMARY)
        verify_close(mean,r["mean_loss"],f"primary mean {m}")
        verify_close(lo,r["bootstrap_ci_95"][0],f"primary bootstrap CI lower {m}")
        verify_close(hi,r["bootstrap_ci_95"][1],f"primary bootstrap CI upper {m}")
        verify_close(float(np.exp(mean)),r["perplexity"],f"primary perplexity {m}")
        primary_models.append({"model":LABELS[m],"n_question_clusters":len(values),"n_source_records":len(values)*4,"mean_full_sequence_loss":mean,"ci95_bootstrap_lower":lo,"ci95_bootstrap_upper":hi,"native_tokenizer_ppl_descriptive":float(np.exp(mean))})
    write_csv(OUTPUT_DIR/"primary_clustered_307_model_summary.csv",primary_models)

    primary_pairs=[]
    for i,a in enumerate(MODELS):
        for j,b in enumerate(MODELS[i+1:],i+1):
            x=np.asarray(p["model_results"][a]["question_scores"],dtype=float)
            y=np.asarray(p["model_results"][b]["question_scores"],dtype=float)
            val=paired_bootstrap_stats(x,y,seed=SEED_PRIMARY+i*101+j,replicates=N_BOOT_PRIMARY,batching=False)
            stored=p["pairwise_statistics"][f"{a}_vs_{b}"]
            verify_close(val["mean_loss_diff"],stored["mean_loss_diff"],f"primary difference {a} vs {b}")
            verify_close(val["ci95_lower"],stored["bootstrap_ci_95"][0],f"primary paired CI lower {a} vs {b}")
            verify_close(val["ci95_upper"],stored["bootstrap_ci_95"][1],f"primary paired CI upper {a} vs {b}")
            verify_close(val["paired_dz"],stored["cohens_dz_paired"],f"primary d_z {a} vs {b}")
            verify_close(val["p_raw"],stored["p_value"],f"primary p {a} vs {b}")
            primary_pairs.append({"model1":LABELS[a],"model2":LABELS[b],"n_questions":len(x),"mean_loss_difference_model1_minus_model2":val["mean_loss_diff"],"ci95_lower":val["ci95_lower"],"ci95_upper":val["ci95_upper"],"paired_dz":val["paired_dz"],"p_raw":val["p_raw"]})
    adjusted=holm([r["p_raw"] for r in primary_pairs])
    for r,padj in zip(primary_pairs,adjusted): r["p_holm"]=padj; r["significant_after_holm_05"]=padj<0.05
    write_csv(OUTPUT_DIR/"primary_clustered_307_pairwise.csv",primary_pairs)

    # Recompute sensitivity model means and CIs using the notebook's batched bootstrap.
    copeland_models=[]
    for m in MODELS:
        r=u["model_results"][m]; values=np.asarray(r["losses"],dtype=float)
        mean=float(values.mean()); lo,hi=bootstrap_ci_batched(values,N_BOOT_COPELAND,SEED_COPELAND)
        verify_close(mean,r["mean_loss"],f"Copeland mean {m}")
        verify_close(lo,r["ci_95_lower"],f"Copeland CI lower {m}")
        verify_close(hi,r["ci_95_upper"],f"Copeland CI upper {m}")
        copeland_models.append({"model":LABELS[m],"n_distinct_questions":len(values),"mean_full_sequence_loss":mean,"ci95_lower":lo,"ci95_upper":hi,"n_windowed_examples":r.get("n_windowed_examples",0),"max_tokenized_sequence_length":r.get("max_tokenized_sequence_length")})
    write_csv(OUTPUT_DIR/"copeland_393_model_summary.csv",copeland_models)

    # Recompute all Copeland pairwise tests using the notebook's single RNG stream and batch algorithm.
    rng=np.random.default_rng(SEED_COPELAND); copeland_pairs=[]
    for i,a in enumerate(MODELS):
        for b in MODELS[i+1:]:
            x=np.asarray(u["model_results"][a]["losses"],dtype=float); y=np.asarray(u["model_results"][b]["losses"],dtype=float); diff=x-y
            t,praw=stats.ttest_rel(x,y,nan_policy="raise"); sd=float(diff.std(ddof=1)); dz=float(diff.mean()/sd) if sd>0 else 0.0
            boot=np.empty(N_BOOT_COPELAND,dtype=float); n=len(diff); batch=500
            for start in range(0,N_BOOT_COPELAND,batch):
                take=min(batch,N_BOOT_COPELAND-start); ix=rng.integers(0,n,size=(take,n)); boot[start:start+take]=diff[ix].mean(axis=1)
            lo,hi=np.percentile(boot,[2.5,97.5])
            stored=u["pairwise_statistics"][f"{a}_vs_{b}"]
            verify_close(diff.mean(),stored["mean_loss_diff"],f"Copeland difference {a} vs {b}")
            verify_close(lo,stored["bootstrap_ci_95"][0],f"Copeland paired CI lower {a} vs {b}")
            verify_close(hi,stored["bootstrap_ci_95"][1],f"Copeland paired CI upper {a} vs {b}")
            verify_close(dz,stored["cohens_dz_paired"],f"Copeland d_z {a} vs {b}")
            verify_close(praw,stored["p_value_raw"],f"Copeland p {a} vs {b}")
            copeland_pairs.append({"model1":LABELS[a],"model2":LABELS[b],"n_questions":len(x),"mean_loss_difference_model1_minus_model2":float(diff.mean()),"ci95_lower":float(lo),"ci95_upper":float(hi),"paired_dz":dz,"p_raw":float(praw)})
    adjusted=holm([r["p_raw"] for r in copeland_pairs])
    for r,padj in zip(copeland_pairs,adjusted): r["p_holm"]=padj; r["significant_after_holm_05"]=padj<0.05
    write_csv(OUTPUT_DIR/"copeland_393_pairwise.csv",copeland_pairs)

    # Post hoc matched overlap: intentionally exploratory; independently recomputed.
    p_set=set(p_ids); u_prefix={qid[:16]:qid for qid in u_ids}; overlap=sorted(p_set & set(u_prefix))
    if len(overlap)<2: raise ValueError("Insufficient common questions for overlap audit")
    overlap_rows=[]; maps_p={}; maps_u={}; n_overlap=args.bootstrap_overlap
    for idx,m in enumerate(MODELS):
        pmap=p_maps[m]; umap={qid[:16]:v for qid,v in u_maps[m].items()}
        a=np.asarray([pmap[qid] for qid in overlap]); b=np.asarray([umap[qid] for qid in overlap]); diff=b-a
        maps_p[m]=a; maps_u[m]=b
        alo,ahi=bootstrap_ci_batched(a,n_overlap,20261018+idx*11)
        blo,bhi=bootstrap_ci_batched(b,n_overlap,20261019+idx*11)
        dlo,dhi=bootstrap_ci_batched(diff,n_overlap,20261020+idx*11)
        overlap_rows.append({"model":LABELS[m],"n_shared_questions":len(overlap),"four_record_mean":float(a.mean()),"four_record_ci95_lower":alo,"four_record_ci95_upper":ahi,"copeland_mean":float(b.mean()),"copeland_ci95_lower":blo,"copeland_ci95_upper":bhi,"copeland_minus_four_record":float(diff.mean()),"method_difference_ci95_lower":dlo,"method_difference_ci95_upper":dhi})
    write_csv(OUTPUT_DIR/"matched_overlap_method_summary.csv",overlap_rows)
    pair_rows=[]
    for method,maps in [("four_record_question_mean",maps_p),("copeland_single_reference",maps_u)]:
        candidates=[]
        for i,a in enumerate(MODELS):
            for b in MODELS[i+1:]:
                val=paired_bootstrap_stats(maps[a],maps[b],seed=20262000+len(pair_rows)*7,replicates=n_overlap,batching=True)
                candidates.append({"analysis":method,"model1":LABELS[a],"model2":LABELS[b],"n_shared_questions":len(maps[a]),"mean_difference_model1_minus_model2":val["mean_loss_diff"],"ci95_lower":val["ci95_lower"],"ci95_upper":val["ci95_upper"],"paired_dz":val["paired_dz"],"t_statistic":val["t_statistic"],"p_raw":val["p_raw"]})
                pair_rows.append(candidates[-1])
        adjusted=holm([r["p_raw"] for r in candidates])
        for r,padj in zip(candidates,adjusted): r["p_holm_within_analysis_six_contrasts"]=padj; r["significant_after_holm_05"]=padj<0.05
    write_csv(OUTPUT_DIR/"matched_overlap_pairwise.csv",pair_rows)

    source_sha=hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    audit_lines=[
        "# Saved Social Reasoning Results Audit","",
        "This report recomputes statistics from included per-item/per-question losses; it does not load model weights or rerun inference.","",
        "## Data integrity","",f"- Source rows: {len(source_rows)}",f"- Exact question groups: {len(groups)}",f"- Records per question: {dict(group_sizes)}",f"- Source SHA-256: `{source_sha}`",f"- Copeland references: {len(reduced_map)}, all responses match source candidates", "",
        "## Analysis samples","",f"- Primary: {len(p_ids)} question groups, four records per question; {len(p_ids)*4} scored records per checkpoint.",f"- Copeland sensitivity: {len(u_ids)} distinct questions, one reference response per question.",f"- Shared question IDs: {len(overlap)}.","",
        "## Statistical verification","",
        "Primary model means, bootstrap intervals, paired mean differences, paired d_z, paired t-test p-values, and Holm-adjusted p-values were recomputed from the saved question-level scores and checked against the statistics embedded in the result JSON.",
        "The same fields were recomputed for the Copeland analysis from its saved item-level losses and checked against the saved JSON. The matched-overlap analysis is recalculated from the 124 common items and is post hoc/exploratory.","",
        "## Interpretation and provenance","",
        "Both the primary and sensitivity analyses use plain-text question + one space + reference-response serialization and each checkpoint's native tokenizer. The score includes scenario/question and response tokens. The artifacts do not provide independent human validation of social competence or cross-tokenizer calibration. Immutable checkpoint/tokenizer revision hashes and a complete inference environment lockfile are absent, so the audit does not guarantee bit-for-bit replay of model inference.",""
    ]
    (OUTPUT_DIR/"social_reasoning_audit.md").write_text("\n".join(audit_lines),encoding="utf-8")
    print("Audit passed: recomputed primary and Copeland statistics match their saved JSON fields.")
    print(f"Source rows={len(source_rows)}; questions={len(groups)}; Copeland={len(reduced_map)}; primary={len(p_ids)}; sensitivity={len(u_ids)}; overlap={len(overlap)}")
    print(f"CSV files and audit report written to {OUTPUT_DIR.relative_to(ROOT).as_posix()}")

if __name__=="__main__":
    main()
