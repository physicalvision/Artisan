<p align="center">
  <img src="../assets/artisan-wordmark.png" width="360" alt="Artisan">
</p>

<h3 align="center"><strong>Benchmark and Evaluation</strong></h3>

<p align="center">
  <a href="https://physicalvision.github.io/artisan"><img src="https://img.shields.io/badge/Project-Page-orange" alt="Project Page"></a> &nbsp;&nbsp;&nbsp;
  <a href="https://huggingface.co/datasets/mastercai/Artisan-Benchmark"><img src="https://img.shields.io/badge/HuggingFace-Dataset-yellow" alt="Hugging Face Dataset"></a>
</p>

> Generate articulated assets and evaluate their appearance, geometry, materials, collisions, stability and mass with **Artisan-Benchmark**.

Complete the installation and API key setup in the [main README](../README.md),
then run all commands below from the project root.

## 📦 Dataset Preparation

Download [Artisan-Benchmark](https://huggingface.co/datasets/mastercai/Artisan-Benchmark)
and place the four dataset folders under `benchmark/data/`:

```text
benchmark/data/
├── ArtVIP/
├── LightWheel/
├── Stability/
└── ABO-Mass/
```

| Dataset | Generation inputs |
| --- | --- |
| ArtVIP | Text and one open-state reference image |
| LightWheel | Text and one open-state reference image |
| Stability | Text only |
| ABO-Mass | Text and one reference image |

## 🔥 Asset Generation

Generate assets for a full dataset:

```bash
.artisan/venv/bin/python benchmark/run.py --dataset ArtVIP --all
.artisan/venv/bin/python benchmark/run.py --dataset LightWheel --all
.artisan/venv/bin/python benchmark/run.py --dataset Stability --all
.artisan/venv/bin/python benchmark/run.py --dataset ABO-Mass --all
```

Use `--case OBJECT_ID` for selected cases or `--all` for the full dataset.
Inputs come from `benchmark/data/`.

## 📊 Evaluation

### 🎨 Appearance, Geometry and Materials

Evaluate ArtVIP assets with the following commands. Replace `ArtVIP` with
`LightWheel` to evaluate that dataset.

```bash
.artisan/venv/bin/python benchmark/Evaluation/Fidelity/evaluate.py --dataset ArtVIP --mode 2d
.artisan/venv/bin/python benchmark/Evaluation/Fidelity/evaluate.py --dataset ArtVIP --mode 3d
.artisan/venv/bin/python benchmark/Evaluation/Fidelity/evaluate.py --dataset ArtVIP --mode material_direct
.artisan/venv/bin/python benchmark/Evaluation/Fidelity/evaluate.py --dataset ArtVIP --mode material_transfer
```

Omit `--mode` or use `--mode all` to evaluate all modes.

### ⚙️ Collisions, Stability and Mass

Evaluate collisions on ArtVIP, stability on Stability, and mass on ABO-Mass:

```bash
.artisan/venv/bin/python benchmark/Evaluation/Collision/evaluate.py --dataset ArtVIP
.artisan/venv/bin/python benchmark/Evaluation/Stability/evaluate.py
.artisan/venv/bin/python benchmark/Evaluation/Mass/evaluate.py
```

Metric tables are printed and saved as JSON and CSV in the method directory.

## 📁 Output Structure

Generation creates `native/`; evaluation adds poses and scores. For example:

```text
outputs/ArtVIP/METHOD/
├── OBJECT_ID/
│   ├── native/
│   ├── eval_poses/
│   └── evaluation/
├── summary_MODE.*
└── summary_collision.*
```

## 📚 Documentation

- [Installation, configuration and visualization](../README.md)
- [Configuration example](../artisan.example.toml)
- [SDK model contract](../sdk/_docs/model_contract.md)
