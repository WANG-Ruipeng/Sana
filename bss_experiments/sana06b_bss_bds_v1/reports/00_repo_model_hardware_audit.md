| Item | Found? | Path / Evidence | Notes | Action |
| --- | --- | --- | --- | --- |
| current repo path | yes | E:\Sana |  | none |
| git remote | yes | origin	https://github.com/WANG-Ruipeng/Sana.git (fetch); origin	https://github.com/WANG-Ruipeng/Sana.git (push); upstream	https://github.com/NVlabs/Sana.git (fetch); upstream	https://github.com/NVlabs/Sana.git (push) |  | none |
| current branch | yes | sana06b-bss-bds |  | none |
| commit hash | yes | 59629fdf790850797cb657bad014fce432bd713d |  | none |
| dirty status | no | ?? Prompt.txt<br>?? bss_experiments/ |  | review before commit |
| Python version | yes | 3.14.2 | C:\Users\warpwang\AppData\Local\Python\pythoncore-3.14-64\python.exe | none |
| PyTorch version | no | ModuleNotFoundError: No module named 'torch' |  | install torch in Colab if missing |
| CUDA version | no | torch unavailable |  | use Colab GPU runtime for experiment |
| nvidia-smi | yes | NVIDIA GeForce RTX 5080, 16303 MiB |  | use GPU runtime if unavailable |
| GPU name and VRAM | no | none; unknown |  | use >=16GB if possible |
| running in Colab | no | False |  | intended runner is Colab |
| Drive mounted | no | /content/drive/MyDrive |  | mount Drive in notebook |
| Sana official repo exists locally | yes | E:\Sana | current checkout is a Sana repo/fork | none |
| app/sana_pipeline.py exists | yes | E:\Sana\app\sana_pipeline.py |  | none |
| Sana_600M_img1024.yaml exists | yes | E:\Sana\configs\sana_config\1024ms\Sana_600M_img1024.yaml |  | none |
| Sana_1600M_img1024.yaml exists | yes | E:\Sana\configs\sana_config\1024ms\Sana_1600M_img1024.yaml |  | none |
| diffusers import | no | ModuleNotFoundError: No module named 'diffusers' |  | pip install diffusers>=0.32.0 if missing |
| diffusers SanaPipeline usable | no | ModuleNotFoundError: No module named 'diffusers' |  | native remains primary |
| diffusers>=0.32.0 | no | ModuleNotFoundError: No module named 'diffusers' |  | upgrade if version is below 0.32.0 |
| native SanaPipeline import | no | ModuleNotFoundError: No module named 'pyrallis' | from app.sana_pipeline import SanaPipeline | install repo deps if missing |
| HF pth model repo id | yes | Efficient-Large-Model/Sana_600M_1024px | primary native checkpoint candidate | audit model card before run |
| HF diffusers model repo id | yes | Efficient-Large-Model/Sana_600M_1024px_diffusers | fallback candidate listed in local docs | do not force if unavailable |
| Drive weights for Sana_600M_1024px | no | \content\drive\MyDrive\ModelWeights\Sana\Sana_600M_1024px | weights must stay outside git | download to Drive if missing |

## Model mapping note

Local Sana docs list both the pth repo `Efficient-Large-Model/Sana_600M_1024px` and diffusers repo `Efficient-Large-Model/Sana_600M_1024px_diffusers` for Sana-0.6B 1024px. The selected native config is `configs/sana_config/1024ms/Sana_600M_img1024.yaml`, whose model string is `SanaMS_600M_P1_D28` and flow shift is 4.0. The runner still validates the checkpoint path at load time.
## Missing weights action

Weights were not found locally. Download them to Google Drive, not this repo:

```bash
huggingface-cli download Efficient-Large-Model/Sana_600M_1024px --local-dir "$DRIVE_WEIGHTS_ROOT/Sana_600M_1024px"
```

## Backend selection

Backend A (`native`) is selected as primary for this experiment because it uses `app.sana_pipeline.SanaPipeline` with the local 600M config and the official Flow-DPM-Solver path. Backend B (`diffusers`) is available only if the 0.6B diffusers checkpoint loads and exposes compatible timestep control for BSS.
