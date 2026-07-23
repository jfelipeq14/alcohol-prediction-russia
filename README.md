# Alcohol Consumption Prediction in Russia (2024)

Deep neural network to predict pure alcohol consumption per capita in Russian regions for 2024, using historical data from 2017-2023.

## Problem Statement

Supervised regression task. The model learns from 7 years of historical data (2017-2023) across ~85 Russian regions and 6 beverage types to forecast consumption for 2024.

**Target variable**: Consumption of alcoholic beverages (in liters of pure alcohol per capita)

## Tech Stack

- **Language**: Python 3.10
- **Framework**: PyTorch
- **Data**: pandas, numpy, scikit-learn
- **Visualization**: matplotlib, seaborn
- **Containerization**: Docker

## Project Structure

```
├── src/
│   ├── data/          → Loading, cleaning, pivot, splits
│   ├── model/         → Neural network architecture
│   ├── training/      → Training loop, early stopping
│   ├── evaluate/      → Metrics (MSE, MAE, R²)
│   └── predict/       → 2024 forecast generation
├── notebooks/         → Exploration & visualization
├── data/              → Original dataset (CSV)
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Quick Start

### With Docker (recommended)

```bash
# Build the image
docker compose build

# Run the full pipeline
docker compose run app python src/main.py

# Open a shell inside the container
docker compose run app bash
```

### Without Docker

```bash
pip install -r requirements.txt
python src/main.py
```

## Dataset

The dataset contains annual consumption data of alcoholic beverages across Russian regions (2017-2023), with 6 beverage types: Wine, Beer, Vodka, Sparkling wine, Brandy, and Cider.

## License

MIT
