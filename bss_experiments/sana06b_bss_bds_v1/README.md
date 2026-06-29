# Sana-0.6B BSS/BDS experiment

This folder contains the lightweight text-to-image BSS/BDS sanity experiment for
Sana-0.6B / `Efficient-Large-Model/Sana_600M_1024px`.

The code is designed so that GitHub stores the experiment logic, manifests,
notebook, and small reports. Model weights, generated images, runtime outputs,
metrics, and large artifacts should live on Google Drive or another external
artifact store.

Primary runner:

```bash
python bss_experiments/sana06b_bss_bds_v1/scripts/audit_sana06b.py
python bss_experiments/sana06b_bss_bds_v1/scripts/make_manifest_sana06b_bds.py
python bss_experiments/sana06b_bss_bds_v1/scripts/run_manifest.py \
  --manifest bss_experiments/sana06b_bss_bds_v1/manifests/sana06b_smoke_manifest.csv \
  --resume
```

Use the Colab notebook for the intended GPU/Drive workflow.
