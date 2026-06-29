# Sana-0.6B smoke report

Status: not run in this local workspace.

Smoke requires a GPU runtime and Sana-0.6B weights outside git, normally on Google Drive. Run the Colab notebook and then regenerate metrics/reports after the smoke manifest completes.

Required smoke rows:

- `uniform8`
- `uniform10`
- `bss10`
- `reference_uniform50`

Validation gates:

- output image exists for every completed row
- schedule JSON exists for every completed row
- `uniform8`, `uniform10`, and `reference_uniform50` eval counts match 8, 10, and 50
- `bss10` has 10 evals from `base8 + split first/last`
- `bss10` schedule differs from `uniform10`

Verdict: Need more data.
