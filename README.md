# Deep Learning Portfolio

A curated collection of reproducible deep-learning projects built around clear
problem definitions, honest evaluation, tested code, and deployment-ready
artifacts.

## Featured projects

| Project | Domain | Architecture | Highlights | Status |
|---|---|---|---|---|
| [KMD Weather AI](projects/kmd-weather-ai/) | Spatiotemporal weather forecasting | Temporal-Attention U-Net | Multi-horizon quantiles, rain probability, persistence baselines, tiled NetCDF inference | Research prototype |

## Engineering standards

Every portfolio project is expected to include:

- a concise problem statement and explicit data contract;
- an architecture whose design choices are explained;
- reproducible environments and deterministic experiment settings;
- baselines and metrics appropriate to the problem;
- automated tests for the reusable pipeline;
- documented limitations and responsible-use boundaries; and
- no committed credentials, personal information, restricted data, model
  checkpoints, or machine-specific notebook output.

## Technical focus

- PyTorch model design and multi-task learning
- spatiotemporal and probabilistic forecasting
- data validation and leakage-aware experiment design
- scientific evaluation and uncertainty quantification
- memory-aware inference and portable artifact export
- tested, package-oriented notebook workflows

## Repository structure

```text
deep-learning-portfolio/
├── .github/workflows/          automated quality checks
├── projects/
│   └── kmd-weather-ai/         probabilistic WRF forecasting project
├── CONTRIBUTING.md             checklist for future portfolio projects
└── README.md                   portfolio index
```

Each project is self-contained. Open its README for data requirements, setup,
scientific caveats, and validation instructions.

## Run the current project

```bash
cd projects/kmd-weather-ai
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pytest
```

Raw datasets and generated artifacts are intentionally kept outside Git. This
repository contains only publishable source code, tests, documentation, and
sanitized notebooks.
