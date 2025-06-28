# Distributed ML Platform - Fully Integrated System

A complete distributed machine learning platform built with Ray, FastAPI, Docker, and Streamlit. This system provides end-to-end distributed model training, serving, and visualization capabilities.

## 🏗️ Architecture

- **Backend (Ray + FastAPI)**: Distributed model training and serving
- **Frontend (Streamlit)**: User interface for cluster management and training
- **Containerization (Docker)**: Fully containerized deployment
- **Visualization**: Real-time metrics and model performance charts

## ✨ Features

### 🎯 Complete Integration
- ✅ **No TODOs or Legacy Code**: Fully implemented system
- ✅ **End-to-End Workflow**: From data upload to model serving
- ✅ **Distributed Training**: Multi-model parallel training with Ray
- ✅ **Real-time Visualization**: ROC curves, learning curves, metrics
- ✅ **Cluster Management**: Ray cluster monitoring and control
- ✅ **Model Serving**: Production-ready API for predictions

## 🏗️ Architecture

```
distributed-ml-platform/
│
├── interface/
│   └── interface.py          # Streamlit UI (main entry point)
├── main.py                   # Model serving backend
├── docker-compose.yml        # Single Streamlit service
│
└── src/                      # Backend modules
    ├── data/                 # Simplified data processing
    │   └── data_loader.py
    ├── models/               # Distributed model training
    │   └── model_trainer.py
    ├── serving/              # API with cluster management
    │   └── api.py
    ├── utils/                # Ray utilities
    │   └── ray_utils.py
    └── visualization/        # Plotting and dashboards
        └── visualizer.py
```

## ✨ New Workflow

1. **Start**: `docker-compose up -d`
2. **Access UI**: http://localhost:8501
3. **Configure Cluster**: Set head node and worker resources in sidebar
4. **Deploy Ray**: Click "Deploy Ray" to start distributed cluster
5. **Start API**: Click "Serve API" to enable model serving
6. **Train Models**: Upload datasets and configure training in UI
7. **Monitor**: Real-time status updates and logs
8. **Use Models**: Access trained models via API endpoints

## 🛠️ Technology Stack

- **Frontend**: Streamlit (Web UI)
- **Backend**: FastAPI (RESTful API)
- **Distributed Computing**: Ray (Cluster management)
- **ML Libraries**: Scikit-Learn, Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Containerization**: Docker, Docker Compose

## 🚀 Quick Start

### 1. Deploy the Platform
```bash
docker-compose up -d
```

### 2. Access the Interface
Open your browser and go to: **http://localhost:8501**

### 3. Configure Ray Cluster
- In the sidebar, configure head node resources (CPU/RAM)
- Add workers using "New Worker" button
- Set resources for each worker
- Click "Deploy Ray" to start the cluster

### 4. Start API Server
- Once Ray is running, click "Serve API"
- API will be available at: **http://localhost:8000**

### 5. Train Models
- Go to "Training" section in the UI
- Upload datasets or use sample data
- Configure ML tasks (Classification/Regression)
- Select target columns
- Click "Start Training"

### 6. Monitor & Use
- Monitor cluster status in real-time
- View logs in the sidebar
- Access model predictions via API
- View visualizations and dashboards

## 📊 API Endpoints

### Core Endpoints
- `GET /health` - API health check
- `GET /models` - List available models
- `GET /datasets` - List available datasets
- `GET /cluster/status` - Ray cluster status

### Prediction Endpoints
- `POST /predict/{model_name}` - Predict with specific model
- `POST /predict/all/{dataset}` - Predict with all models for a dataset
- `POST /predict/all` - Predict with all available models

### Metrics & Visualization
- `GET /metrics/{model_name}` - Get model metrics
- `GET /metrics/all` - Get all model metrics
- `GET /visualization/{model_name}/roc` - ROC curve (PNG)
- `GET /visualization/{model_name}/learning_curve` - Learning curve (PNG)
- `GET /visualization/all` - Complete dashboard (HTML)

### Training & Management
- `POST /train` - Start distributed training
- `POST /cluster/add_worker` - Add cluster worker
- `POST /cluster/remove_worker` - Remove cluster worker

## 🔧 Configuration

All configuration is done through the Streamlit UI:

- **Head Node**: 1-8 CPUs, 1-16 GB RAM
- **Workers**: Up to 5 workers, each with 1-6 CPUs, 1-8 GB RAM
- **Training**: Custom datasets, task types, target columns
- **Monitoring**: Real-time resource usage and logs

## 📁 Data Management

- **No Volumes/Shared Folders**: Fully distributed, no external dependencies
- Sample datasets baked into container image at build time
- Upload custom datasets via Streamlit UI (processed in-memory)
- Models stored in Ray's distributed memory (not on disk)
- Ray actors maintain model state across distributed nodes
- Visualizations generated on-demand from in-memory models

## 🐳 Container Architecture

- **Single Service**: Streamlit interface handles everything
- **Dynamic Ray Deployment**: Cluster created on-demand from UI
- **API Integration**: FastAPI server started from Streamlit
- **Resource Isolation**: Ray processes isolated within container
- **Persistent Storage**: Models and outputs saved to container storage

## 🔍 Monitoring & Debugging

### Access Points
- **Streamlit UI**: http://localhost:8501
- **API (when active)**: http://localhost:8000  
- **Ray Dashboard (when active)**: http://localhost:8265

### Logs
```bash
# View container logs
docker-compose logs streamlit-interface

# Follow logs in real-time
docker-compose logs -f streamlit-interface
```

### Troubleshooting
- **UI not loading**: Check container logs
- **Ray cluster issues**: Verify resource allocation in UI
- **Training failures**: Check dataset format and target columns
- **API not responding**: Ensure Ray cluster is running first

## 🆕 Migration from Legacy

This version completely replaces the previous hardcoded Docker Compose setup:

- ❌ **Legacy**: Hardcoded ray-head, ray-worker-1, ray-worker-2, trainer, api-server services
- ✅ **New**: Single Streamlit service with dynamic Ray cluster management
- ❌ **Legacy**: Automatic training on startup  
- ✅ **New**: User-controlled, on-demand training
- ❌ **Legacy**: Fixed resource allocation
- ✅ **New**: Dynamic, user-configurable resources
- ❌ **Legacy**: Command-line driven workflow
- ✅ **New**: Complete web-based interface

## 📝 Example Usage

### Training Custom Models
1. Access http://localhost:8501
2. Configure and deploy Ray cluster
3. Upload your CSV dataset
4. Select classification or regression task
5. Choose target column
6. Start training and monitor progress

### Making Predictions
```python
import requests

# Single model prediction
response = requests.post(
    "http://localhost:8000/predict/RandomForestClassifier_iris",
    json={"features": [{"sepal_length": 5.1, "sepal_width": 3.5, 
                       "petal_length": 1.4, "petal_width": 0.2}]}
)
print(response.json())

# All models for a dataset
response = requests.post(
    "http://localhost:8000/predict/all/iris",
    json={"features": [{"sepal_length": 5.1, "sepal_width": 3.5, 
                       "petal_length": 1.4, "petal_width": 0.2}]}
)
print(response.json())
```

---

**🎯 This platform provides a complete, user-friendly solution for distributed machine learning with modern web-based controls and real-time monitoring capabilities.**

- This deployment does **not** use Docker volumes. All coordination is handled by Ray over the network.
- Data loss may occur if a container is deleted. For persistence, consider uploading results to a remote location.
