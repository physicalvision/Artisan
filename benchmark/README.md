# Benchmarks

Run commands from the project root.

Download the data from [Artisan-Benchmark](https://huggingface.co/datasets/mastercai/Artisan-Benchmark) and place the four dataset folders under `benchmark/data/`:

```text
benchmark/data/
  ArtVIP/
  LightWheel/
  Stability/
  ABO-Mass/
```

## Generate

```bash
.artisan/venv/bin/python benchmark/run.py --dataset ArtVIP --all
.artisan/venv/bin/python benchmark/run.py --dataset LightWheel --all
.artisan/venv/bin/python benchmark/run.py --dataset Stability --all
.artisan/venv/bin/python benchmark/run.py --dataset ABO-Mass --all
```

Use `--case OBJECT_ID` for selected cases or `--all` for the full dataset.
ArtVIP and LightWheel use one open-state reference image and text; Stability uses text only;
ABO-Mass uses one reference image and text. Inputs come from `benchmark/data/`.

Generation creates `native/`; evaluation adds poses and scores:

```text
outputs/ArtVIP/METHOD/
  OBJECT_ID/
    native/
    eval_poses/
    evaluation/
  summary_MODE.*
  summary_collision.*
```

## Appearance, geometry and material

For ArtVIP or LightWheel:

```bash
.artisan/venv/bin/python benchmark/Evaluation/Fidelity/evaluate.py --dataset ArtVIP --mode 2d
.artisan/venv/bin/python benchmark/Evaluation/Fidelity/evaluate.py --dataset ArtVIP --mode 3d
.artisan/venv/bin/python benchmark/Evaluation/Fidelity/evaluate.py --dataset ArtVIP --mode material_direct
.artisan/venv/bin/python benchmark/Evaluation/Fidelity/evaluate.py --dataset ArtVIP --mode material_transfer
```

Omit `--mode` or use `--mode all` to evaluate all modes.


## Collision, stability and mass

```bash
.artisan/venv/bin/python benchmark/Evaluation/Collision/evaluate.py --dataset ArtVIP
.artisan/venv/bin/python benchmark/Evaluation/Stability/evaluate.py
.artisan/venv/bin/python benchmark/Evaluation/Mass/evaluate.py
```

Metric tables are printed and saved as JSON and CSV in the method directory.

