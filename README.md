# Alcohol Consumption Prediction in Russia (2024)

Deep neural network to predict pure alcohol consumption per capita in Russian regions for 2024, using historical data from 2017-2023.

**Test performance**: R² > 0.99 | MSE ~0.01 | MAE ~0.06

## Quick Start

### Docker (recommended)
```bash
docker compose run --remove-orphans app
```

### Local
```bash
pip install -r requirements.txt
python src/main.py
```

### Google Colab
1. Go to https://colab.research.google.com
2. File → Upload notebook → select `colab.ipynb`
3. Runtime → Run all (no setup required)

## Output

The pipeline generates the following files in `output/`:

| File | Description |
|---|---|
| `training_history.png` | Train/val loss curves with best epoch marker |
| `actual_vs_predicted.png` | Side-by-side NN vs Linear Regression scatter |
| `ranking.png` | Top 15 and bottom 15 regions by predicted consumption |
| `model_comparison.png` | NN vs LR: MSE, MAE, R² bar chart |
| `beverage_ranking.png` | Consumption ranking by beverage type |
| `comparison_2023_2024.png` | 2023 actual vs 2024 predicted scatter |
| `residuals.png` | Residual distribution, scatter, and boxplot |
| `correlation_heatmap.png` | Year-to-year correlation heatmap |
| `predicciones_2024.csv` | Full predictions (595 rows: 85 regions × 7 beverages) |
| `ranking_regiones.csv` | Region-level aggregated ranking |
| `ranking_bebidas.csv` | Beverage-level aggregated ranking |
| `best_model.pth` | Trained model checkpoint |

## Dataset

- **4,165 rows**, 85 Russian regions, **7 beverage types** (Wine, Beer, Vodka, Sparkling wine, Brandy, Cider, Liqueurs)
- **Years**: 2017–2023
- **Encoding**: cp1252 (legacy Windows); `�ider` is normalized to `Cider` via regex
- **Source file**: `Consumption of alcoholic beverages 2017-2023 (Pivot table).csv`

## Model Architecture

```
Input (110 features) → Linear(110, 128) → ReLU → Dropout(0.3)
                     → Linear(128, 64)  → ReLU → Dropout(0.3)
                     → Linear(64, 1)
```

- **22,529 parameters**
- **Loss**: MSELoss | **Optimizer**: Adam (lr=0.001, weight_decay=1e-5)
- **Batch**: 32 | **Early stopping**: patience 15 (restores best weights)
- **Input features**: 18 time-series (3 metrics × 6 years) + `alc_2023` + one-hot region + one-hot beverage

## Project Structure

```
├── src/
│   ├── data/          → loader, preprocessor, splitter, dataset
│   ├── model/         → AlcoholPredictor architecture
│   ├── training/      → training loop, early stopping
│   ├── evaluate/      → metrics (MSE, MAE, R²), sklearn baseline
│   └── predict/       → 2024 forecast, region/beverage rankings
├── colab.ipynb        → self-contained notebook for Google Colab
├── AGENTS.md          → detailed implementation guide for collaborators
├── assets/            → generated images for the project report
├── Informe*.md        → complete analysis report (Spanish)
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Contributing

See [AGENTS.md](AGENTS.md) for the full implementation guide, conventions, and ethical considerations.

Key guidelines:
- Do not add libraries without consulting
- Keep preprocessing separate from model architecture
- Verify the pipeline runs before committing
- Do not commit without explicit authorization

## License

MIT
