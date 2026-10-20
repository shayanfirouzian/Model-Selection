#!/usr/bin/env python3
"""Generate figures from the saved primary and sensitivity Social Reasoning result JSONs.

No model inference is performed. Outputs are generated from saved losses and
statistics under results/social_reasoning/.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
PRIMARY=ROOT/"results/social_reasoning/primary_clustered_307/social_reasoning_results.json"
COPELAND=ROOT/"results/social_reasoning/copeland_single_reference_393/social_reasoning_results.json"
OUT=ROOT/"figures/social_reasoning"
ORDER=["Qwen3-8B-Base","llama-3.1-8b-base","llama-3.1-8B-instruct","Qwen3-8B"]
LABELS={
 "Qwen3-8B-Base":"Qwen3 (8B) Base",
 "llama-3.1-8b-base":"Llama 3.1 (8B) Base",
 "llama-3.1-8B-instruct":"Llama 3.1 (8B) Instruct",
 "Qwen3-8B":"Qwen3 (8B) Instruct",
}


def plot_ci(data_path:Path, name:str, ci_mode:str, n_label:str):
    d=json.loads(data_path.read_text(encoding="utf-8")); res=d["model_results"]
    means=[res[m]["mean_loss"] for m in ORDER]
    if ci_mode=="bootstrap":
        lo=[res[m]["bootstrap_ci_95"][0] for m in ORDER]; hi=[res[m]["bootstrap_ci_95"][1] for m in ORDER]
    else:
        lo=[res[m]["ci_95_lower"] for m in ORDER]; hi=[res[m]["ci_95_upper"] for m in ORDER]
    err=np.array([[means[i]-lo[i] for i in range(len(ORDER))],[hi[i]-means[i] for i in range(len(ORDER))]])
    fig,ax=plt.subplots(figsize=(8.4,5.3))
    x=np.arange(len(ORDER)); ax.bar(x,means,yerr=err,capsize=5)
    ax.set_xticks(x,[LABELS[m] for m in ORDER],rotation=25,ha="right")
    ax.set_ylabel("Mean full-sequence loss (lower is lower loss)")
    ax.set_title(f"Social Reasoning full-sequence loss ({n_label})")
    ax.grid(axis="y",alpha=0.25); fig.tight_layout()
    path=OUT/name; fig.savefig(path,dpi=300,bbox_inches="tight"); plt.close(fig)
    print(path.relative_to(ROOT).as_posix())


def plot_effects():
    d=json.loads(PRIMARY.read_text(encoding="utf-8")); results=d["pairwise_statistics"]
    matrix=np.zeros((len(ORDER),len(ORDER)),dtype=float)
    for value in results.values():
        a=value["model1"]; b=value["model2"]
        dz=float(value["cohens_dz_paired"])
        i,j=ORDER.index(a),ORDER.index(b)
        matrix[i,j]=dz; matrix[j,i]=-dz
    fig,ax=plt.subplots(figsize=(8.0,6.0))
    lim=max(float(np.abs(matrix).max()),1e-9)
    im=ax.imshow(matrix,cmap="coolwarm",vmin=-lim,vmax=lim)
    ax.set_xticks(range(len(ORDER)),[LABELS[m] for m in ORDER],rotation=35,ha="right")
    ax.set_yticks(range(len(ORDER)),[LABELS[m] for m in ORDER])
    ax.set_title("Paired standardized effects (Cohen's $d_z$): primary 307-question analysis")
    for i in range(len(ORDER)):
        for j in range(len(ORDER)):
            ax.text(j,i,"—" if i==j else f"{matrix[i,j]:+.2f}",ha="center",va="center",fontsize=9)
    fig.colorbar(im,ax=ax,label="Paired $d_z$; positive = row has higher loss")
    fig.tight_layout(); path=OUT/"primary_clustered_307_effect_sizes.png"
    fig.savefig(path,dpi=300,bbox_inches="tight"); plt.close(fig); print(path.relative_to(ROOT).as_posix())


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    plot_ci(PRIMARY,"primary_clustered_307_loss_ci.png","bootstrap","307 question clusters")
    plot_ci(COPELAND,"copeland_single_reference_393_loss_ci.png","item","393 distinct questions")
    plot_effects()

if __name__=="__main__":
    main()
