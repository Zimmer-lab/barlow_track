# Revision analyses (paper resubmission)

Scripts that produce results for the response to the reviewers. Test numbers follow `review/TEST_PLAN.md` in
the paper repository (BarlowTrack-Zero-shot-multi-object-cell-tracking-using-modified-Barlow-Twins). Outputs go
to `/lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_revisions/revision_analysis/` (T1, T2) or into
`config_files/round_shared_v1/` (T3), not into the repo.

| Test | Script | Question | Runs on |
|---|---|---|---|
| shared | `gt_positions.py` | GT positions (cached), detection NN distance, accuracy from saved tracks | – |
| T1 | `t1_deformation_table.py` | Deformation per dataset; accuracy vs per-frame deformation (R3.W1/W3) | local CPU |
| T1 videos | `export_supplementary_videos.py` | How do predicted neuron identities hold up during visible deformation? | local CPU, after T1 |
| T2 | `t2_modality_contribution.py` | Where image crops add over position-only (R1.M5, R2.1, R3.W4) | local CPU, after T1 |
| T3 | `t3_launch.sh`, `t3_report.py` | Per-time exclusivity ablation and a fair HDBSCAN (R2.3) | cluster GPU jobs |

The training and re-tracking machinery these rely on lives elsewhere: the sensitivity rounds
(`scratch/round_sens_src/`, including `grid_posonly_reps.yaml` for T2) and `scratch/gt_free_proxy/rescore_round.py`
(T3).

## Supplementary videos

Each of the 12 benchmark recordings has a standard MP4 and a debug copy with arrows from incorrect predictions to known reference positions. See the [viewing and export guide](SUPPLEMENTARY_VIDEOS.md) for the shared folder, marker legend, accuracy definition, and a worked example.
