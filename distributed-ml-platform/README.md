# Distributed ML Platform

A distributed machine learning platform using Ray, Docker, and Scikit-Learn for parallel training and serving of machine learning models.

GitHub Repository: [https://github.com/DJSixHub/scr_pasd_2025](https://github.com/DJSixHub/scr_pasd_2025)

## Project Structure

```
distributed-ml-platform/
│
├── main.py               # Main entry point
├── requirements.txt      # Python dependencies
├── Dockerfile            # Docker container definition
├── docker-compose.yml    # Docker Compose configuration for distributed deployment
│
└── src/                  # Source code
    ├── data/             # Data loading and preprocessing
    │   └── data_loader.py
    │
    ├── models/           # Model training and evaluation
    │   └── model_trainer.py
    │
    ├── serving/          # Model serving API
    │   └── api.py
    │
    ├── utils/            # Utility functions
    │   └── ray_utils.py
    │
    └── visualization/    # Visualization utilities
        └── visualizer.py
```

## Features

- Distributed data loading and preprocessing
- Parallel training of multiple ML models
- Model serving via API
- Performance visualization and monitoring
- Distributed architecture with Ray
- Docker containerization for easy deployment

## Getting Started

### Prerequisites

- Python 3.9+
- Docker and Docker Compose (for containerized deployment)
- Ray (for distributed computing)

### Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

### Usage

#### Training Models

Train models on single or multiple datasets:

```
python main.py --mode train --data path/to/dataset1.csv path/to/dataset2.csv --target target_column
```

#### Serving Models

Start the API server with trained models (no host volumes required):

```
docker compose up
```

#### Training and Serving

Perform both operations (all data is available inside the container):

```
docker compose up
```

### Docker Deployment

Deploy the distributed platform with Docker Compose (no host volumes):

```
docker compose up
```

## API Endpoints

- `GET /health` - Check API health
- `GET /models` - List available models
- `POST /predict/<model_name>` - Make predictions with a specific model
- `GET /metrics` - Get serving metrics

## Data and Output

- All datasets are included in the image at build time.
- All output (models, plots, metrics) is stored inside the container.
- To extract results, use `docker cp` or expose an API endpoint for download.

## Note on Volumes

- This deployment does **not** use Docker volumes. All coordination is handled by Ray over the network.
- Data loss may occur if a container is deleted. For persistence, consider uploading results to a remote location.
