# Missing / Skipped Figures

The following figures could not be generated because the underlying data is absent from the repository:

- **fig_01_pretrain_loss.png — NO pretraining trainer_state.json or training log found (no models/*_text_* dir, no text-stage log).**
- **fig_02_qa_loss.png — NO QA-stage trainer_state.json or training log found (no models/*_qa_* dir, no qa-stage log).**
- **fig_06_perplexity_comparison.png — NO perplexity values recorded (pretraining never run/logged; compute_metrics exists but no eval output saved).**
- **fig_07_em_f1_comparison.png — NO EM/F1 scores found for any model version (eval scripts exist in test/ but no saved results/outputs).**

## Figures successfully generated

- fig_03_dpo_loss.png
- fig_03b_dpo_rewards.png
- fig_04_instruction_loss.png
- fig_05_all_losses_combined.png (only stages 3 & 4 have curves; 1 & 2 are placeholders)
- fig_08_token_distribution.png
- fig_09_pipeline_diagram.png
