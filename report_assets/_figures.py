"""Generate report figures into report_assets/figures/ (300 DPI)."""
import os, json, csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

RA = "/home/toru2/Amara/Mongolian_LLM/report_assets"
FIG = os.path.join(RA, "figures")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"figure.dpi": 300, "savefig.dpi": 300, "font.size": 11,
                     "axes.grid": True, "grid.alpha": 0.3, "axes.axisbelow": True})

lh = json.load(open(f"{RA}/data/_log_history.json"))
missing = []


def curve(lh_list):
    tr_s, tr_l, ev_s, ev_l = [], [], [], []
    for e in lh_list:
        if "loss" in e:
            tr_s.append(e["step"]); tr_l.append(e["loss"])
        if "eval_loss" in e:
            ev_s.append(e["step"]); ev_l.append(e["eval_loss"])
    return tr_s, tr_l, ev_s, ev_l


# fig 1: pretrain loss — MISSING
missing.append("fig_01_pretrain_loss.png — NO pretraining trainer_state.json or "
               "training log found (no models/*_text_* dir, no text-stage log).")

# fig 2: qa loss — MISSING
missing.append("fig_02_qa_loss.png — NO QA-stage trainer_state.json or training "
               "log found (no models/*_qa_* dir, no qa-stage log).")

# fig 3: DPO loss
ts, tl, es, el = curve(lh["dpo"])
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(ts, tl, "-o", ms=3, label="Training loss", color="#1f77b4")
ax.plot(es, el, "-s", ms=5, label="Eval loss", color="#d62728")
ax.set_xlabel("Training step"); ax.set_ylabel("Loss")
ax.set_title("Figure 3. DPO Training and Evaluation Loss (Stage 3)")
ax.set_yscale("log"); ax.legend()
fig.tight_layout(); fig.savefig(f"{FIG}/fig_03_dpo_loss.png"); plt.close(fig)

# fig 3b companion: DPO rewards/margins
rs, rc, rr, rm = [], [], [], []
for e in lh["dpo"]:
    if "rewards/chosen" in e:
        rs.append(e["step"]); rc.append(e["rewards/chosen"])
        rr.append(e["rewards/rejected"]); rm.append(e["rewards/margins"])
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(rs, rc, "-o", ms=3, label="rewards/chosen")
ax.plot(rs, rr, "-o", ms=3, label="rewards/rejected")
ax.plot(rs, rm, "-^", ms=3, label="rewards/margins")
ax.set_xlabel("Training step"); ax.set_ylabel("Reward")
ax.set_title("Figure 3b. DPO Rewards and Margins (Stage 3)")
ax.legend()
fig.tight_layout(); fig.savefig(f"{FIG}/fig_03b_dpo_rewards.png"); plt.close(fig)

# fig 4: instruction/search loss
ts4, tl4, es4, el4 = curve(lh["search"])
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(ts4, tl4, "-o", ms=4, label="Training loss", color="#1f77b4")
ax.plot(es4, el4, "-s", ms=6, label="Eval loss", color="#d62728")
ax.set_xlabel("Training step"); ax.set_ylabel("Loss")
ax.set_title("Figure 4. Instruction (Web-Search FC) Training Loss (Stage 4)")
ax.legend()
fig.tight_layout(); fig.savefig(f"{FIG}/fig_04_instruction_loss.png"); plt.close(fig)

# fig 5: 2x2 combined
fig, axs = plt.subplots(2, 2, figsize=(12, 9))
axs[0, 0].text(0.5, 0.5, "Stage 1: Pretraining\n\nNO DATA IN REPO\n(no log / trainer_state)",
               ha="center", va="center", fontsize=12)
axs[0, 0].set_title("Figure 5a. Pretraining Loss"); axs[0, 0].axis("off")
axs[0, 1].text(0.5, 0.5, "Stage 2: QA Fine-tuning\n\nNO DATA IN REPO\n(no log / trainer_state)",
               ha="center", va="center", fontsize=12)
axs[0, 1].set_title("Figure 5b. QA Fine-tuning Loss"); axs[0, 1].axis("off")
axs[1, 0].plot(ts, tl, "-o", ms=2, label="train"); axs[1, 0].plot(es, el, "-s", ms=4, label="eval")
axs[1, 0].set_yscale("log"); axs[1, 0].set_title("Figure 5c. DPO Loss (Stage 3)")
axs[1, 0].set_xlabel("step"); axs[1, 0].set_ylabel("loss"); axs[1, 0].legend()
axs[1, 1].plot(ts4, tl4, "-o", ms=3, label="train"); axs[1, 1].plot(es4, el4, "-s", ms=5, label="eval")
axs[1, 1].set_title("Figure 5d. Instruction Loss (Stage 4)")
axs[1, 1].set_xlabel("step"); axs[1, 1].set_ylabel("loss"); axs[1, 1].legend()
fig.suptitle("Figure 5. Training Losses Across All 4 Stages", fontsize=14)
fig.tight_layout(); fig.savefig(f"{FIG}/fig_05_all_losses_combined.png"); plt.close(fig)

# fig 6: perplexity comparison — MISSING
missing.append("fig_06_perplexity_comparison.png — NO perplexity values recorded "
               "(pretraining never run/logged; compute_metrics exists but no eval output saved).")

# fig 7: EM/F1 comparison — MISSING
missing.append("fig_07_em_f1_comparison.png — NO EM/F1 scores found for any model "
               "version (eval scripts exist in test/ but no saved results/outputs).")

# fig 8: instruction token-length histogram
lens = json.load(open(f"{RA}/data/instruction_token_lengths.json"))
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.hist(lens, bins=40, color="#2ca02c", edgecolor="black", alpha=0.8)
ax.set_xlabel("Token length (Qwen3.5 tokenizer)"); ax.set_ylabel("Number of samples")
ax.set_title("Figure 8. Token-Length Distribution — Instruction Dataset (train, n=%d)" % len(lens))
import statistics as st
ax.axvline(st.mean(lens), color="red", ls="--", label=f"mean={st.mean(lens):.0f}")
ax.legend()
fig.tight_layout(); fig.savefig(f"{FIG}/fig_08_token_distribution.png"); plt.close(fig)

# fig 9: pipeline diagram
fig, ax = plt.subplots(figsize=(12, 3.6))
ax.set_xlim(0, 12); ax.set_ylim(0, 4); ax.axis("off")
stages = [
    ("Stage 1\nContinued\nPretraining\n(CLM, text)", "#aec7e8"),
    ("Stage 2\nQA\nFine-tuning\n(SFT)", "#98df8a"),
    ("Stage 3\nDPO\nAlignment\n(β=0.1)", "#ffbb78"),
    ("Stage 4\nWeb-Search\nFunction-Calling\n(SFT)", "#ff9896"),
]
x = 0.3
for i, (txt, c) in enumerate(stages):
    box = FancyBboxPatch((x, 1.1), 2.4, 1.8, boxstyle="round,pad=0.08",
                         fc=c, ec="black", lw=1.2)
    ax.add_patch(box)
    ax.text(x + 1.2, 2.0, txt, ha="center", va="center", fontsize=10, weight="bold")
    if i < 3:
        ax.add_patch(FancyArrowPatch((x + 2.45, 2.0), (x + 2.95, 2.0),
                                     arrowstyle="-|>", mutation_scale=20, lw=1.5, color="black"))
    x += 2.9
ax.text(0.3, 0.55, "Base: Qwen/Qwen3.5-2B-Base   |   QLoRA (4-bit, r=16, α=16)   |   Unsloth + TRL",
        fontsize=9, style="italic")
ax.text(6, 3.6, "Figure 9. Four-Stage Mongolian LLM Training Pipeline",
        ha="center", fontsize=13, weight="bold")
fig.tight_layout(); fig.savefig(f"{FIG}/fig_09_pipeline_diagram.png"); plt.close(fig)

with open(f"{FIG}/MISSING.md", "w") as f:
    f.write("# Missing / Skipped Figures\n\n")
    f.write("The following figures could not be generated because the underlying "
            "data is absent from the repository:\n\n")
    for m in missing:
        f.write(f"- **{m}**\n")
    f.write("\n## Figures successfully generated\n\n")
    for g in ["fig_03_dpo_loss.png", "fig_03b_dpo_rewards.png",
              "fig_04_instruction_loss.png", "fig_05_all_losses_combined.png "
              "(only stages 3 & 4 have curves; 1 & 2 are placeholders)",
              "fig_08_token_distribution.png", "fig_09_pipeline_diagram.png"]:
        f.write(f"- {g}\n")

print("FIGURES DONE")
print("\n".join(missing))
