import streamlit as st
import requests
import json
import os
import pandas as pd
import base64
import time

st.set_page_config(page_title="Distributed ML Platform Interface", layout="wide")

# --- CLUSTER MANAGEMENT ---
st.sidebar.title("Menú principal")
section = st.sidebar.radio("Selecciona una sección", ["Cluster", "Training", "Modelos y Métricas", "Predicción"])

if 'cluster' not in st.session_state:
    st.session_state['cluster'] = {
        'head': {'cpu': 2, 'ram': 4, 'running': False},
    }

def get_cluster_status():
    """Get cluster status from backend API"""
    try:
        response = requests.get('http://localhost:8000/cluster/status', timeout=5)
        if response.status_code == 200:
            return response.json()
        return {"error": "Backend unavailable"}
    except Exception as e:
        return {"error": str(e)}

def start_head(cpu, ram):
    """In containerized setup, head node is managed by docker-compose"""
    st.info("En el setup con Docker, el head node se gestiona automáticamente.")
    st.session_state['cluster']['head']['running'] = True
    return True

def stop_head():
    """In containerized setup, head node is managed by docker-compose"""
    st.info("En el setup con Docker, el head node se gestiona automáticamente.")
    st.session_state['cluster']['head']['running'] = False
    return True

def start_worker():
    """Add a real Ray worker container using Docker Compose scaling"""
    try:
        response = requests.post('http://localhost:8000/cluster/add_worker', timeout=30)
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                st.success(f"✅ {result.get('message', 'Worker added successfully')}")
                # Update worker info if available
                if 'ray_cluster_workers' in result:
                    st.info(f"Ray cluster now has {result['ray_cluster_workers']} active workers")
                return True
            else:
                st.error(f"❌ Failed to add worker: {result.get('error', 'Unknown error')}")
                return False
        else:
            st.error(f"❌ Backend error: {response.text}")
            return False
    except Exception as e:
        st.error(f"❌ Error connecting to backend: {e}")
        return False

def stop_worker():
    """Remove a real Ray worker container using Docker Compose scaling"""
    try:
        response = requests.post('http://localhost:8000/cluster/remove_worker', timeout=30)
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                st.success(f"✅ {result.get('message', 'Worker removed successfully')}")
                # Update worker info if available
                if 'ray_cluster_workers' in result:
                    st.info(f"Ray cluster now has {result['ray_cluster_workers']} active workers")
                return True
            else:
                st.error(f"❌ Failed to remove worker: {result.get('error', 'Unknown error')}")
                return False
        else:
            st.error(f"❌ Backend error: {response.text}")
            return False
    except Exception as e:
        st.error(f"❌ Error connecting to backend: {e}")
        return False

if section == "Cluster":
    st.header("Gestión del Clúster Ray Distribuido")
    
    # Get real cluster status from backend
    cluster_status = get_cluster_status()
    
    if "error" not in cluster_status:
        st.subheader("Estado Actual del Clúster")
        
        # Get worker details from backend
        try:
            workers_response = requests.get('http://localhost:8000/cluster/workers', timeout=5)
            worker_details = []
            if workers_response.status_code == 200:
                worker_data = workers_response.json()
                if worker_data.get('success'):
                    worker_details = worker_data.get('workers', [])
        except Exception as e:
            worker_details = []
            st.warning(f"Could not fetch worker details: {e}")
        
        # Create comprehensive cluster table
        st.markdown("### 📋 Nodos del Clúster")
        
        # Prepare table data
        table_data = []
        
        # Add head node
        nodes = cluster_status.get("node_details", [])
        head_node = None
        if nodes:
            head_node = nodes[0]  # First node is typically the head
        
        head_cpu = head_node.get("Resources", {}).get("CPU", 2.0) if head_node else 2.0
        head_memory = head_node.get("Resources", {}).get("memory", 4e9) / 1e9 if head_node else 4.0
        head_status = "🟢 Activo" if head_node and head_node.get("Alive") else "🔴 Inactivo"
        
        table_data.append({
            "Nodo": "🎯 Head Node (ray-head)",
            "CPU": f"{head_cpu}",
            "RAM (GB)": f"{head_memory:.1f}",
            "Estado": head_status,
            "Acciones": "Gestionado automáticamente"
        })
        
        # Add worker nodes
        for worker in worker_details:
            status_icon = "🟢" if worker.get('status') == 'running' else "⏸️" if worker.get('status') == 'exited' else "🔴"
            status_text = {
                'running': 'Activo',
                'exited': 'Pausado', 
                'created': 'Creado',
                'restarting': 'Reiniciando'
            }.get(worker.get('status'), 'Desconocido')
            
            # Extract CPU count from Ray cluster info
            worker_cpu = 2.0  # Default for Docker containers
            worker_memory = 2.0  # Default for Docker containers
            
            # Try to get more accurate resource info from Ray
            for node in nodes[1:]:  # Skip head node
                if node.get("Alive"):
                    worker_cpu = node.get("Resources", {}).get("CPU", 2.0)
                    worker_memory = node.get("Resources", {}).get("memory", 2e9) / 1e9
                    break
            
            table_data.append({
                "Nodo": f"⚙️ Worker {worker['number']} ({worker['name']})",
                "CPU": f"{worker_cpu}",
                "RAM (GB)": f"{worker_memory:.1f}",
                "Estado": f"{status_icon} {status_text}",
                "Acciones": f"Worker #{worker['number']}"
            })
        
        # Display table
        if table_data:
            df = pd.DataFrame(table_data)
            st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Worker Management Section
        st.markdown("---")
        st.subheader("🔧 Gestión de Workers")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Add new worker
            if st.button("➕ Agregar Nuevo Worker", type="primary", use_container_width=True):
                with st.spinner("Agregando nuevo worker..."):
                    if start_worker():
                        st.rerun()
        
        with col2:
            # Individual worker actions
            if worker_details:
                st.markdown("**Acciones por Worker:**")
                
                # Worker selection and actions in columns
                worker_col1, worker_col2, worker_col3 = st.columns([2, 1, 1])
                
                with worker_col1:
                    worker_numbers = [w['number'] for w in worker_details]
                    selected_worker = st.selectbox(
                        "Seleccionar Worker",
                        options=worker_numbers,
                        format_func=lambda x: f"Worker {x}",
                        key="worker_selector"
                    )
                
                with worker_col2:
                    if st.button("⏸️ Pausar", key="pause_worker", use_container_width=True):
                        with st.spinner(f"Pausando Worker {selected_worker}..."):
                            try:
                                response = requests.post(f'http://localhost:8000/cluster/pause_worker/{selected_worker}', timeout=30)
                                if response.status_code == 200:
                                    result = response.json()
                                    if result.get('success'):
                                        st.success(f"✅ Worker {selected_worker} pausado exitosamente")
                                        st.rerun()
                                    else:
                                        st.error(f"❌ Error: {result.get('error')}")
                                else:
                                    st.error(f"❌ Error del servidor: {response.text}")
                            except Exception as e:
                                st.error(f"❌ Error de conexión: {e}")
                
                with worker_col3:
                    if st.button("🗑️ Eliminar", key="delete_worker", use_container_width=True):
                        with st.spinner(f"Eliminando Worker {selected_worker}..."):
                            try:
                                response = requests.post(f'http://localhost:8000/cluster/delete_worker/{selected_worker}', timeout=30)
                                if response.status_code == 200:
                                    result = response.json()
                                    if result.get('success'):
                                        st.success(f"✅ Worker {selected_worker} eliminado exitosamente")
                                        st.rerun()
                                    else:
                                        st.error(f"❌ Error: {result.get('error')}")
                                else:
                                    st.error(f"❌ Error del servidor: {response.text}")
                            except Exception as e:
                                st.error(f"❌ Error de conexión: {e}")
            else:
                st.info("No hay workers disponibles para gestionar")
        
        # Cluster summary metrics
        st.markdown("---")
        st.subheader("📊 Métricas del Clúster")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Nodos Totales", cluster_status.get("nodes", 0))
        with col2:
            st.metric("CPUs Totales", cluster_status.get("cluster_resources", {}).get("CPU", 0))
        with col3:
            st.metric("CPUs Disponibles", cluster_status.get("available_resources", {}).get("CPU", 0))
        with col4:
            st.metric("Memoria Total (GB)", round(cluster_status.get("cluster_resources", {}).get("memory", 0) / 1e9, 2))
        
        # Resource utilization
        if "summary" in cluster_status and cluster_status["summary"]:
            cpu_util = cluster_status["summary"].get("cpu_utilization", 0)
            memory_util = cluster_status["summary"].get("memory_utilization", 0)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Uso de CPU", f"{cpu_util:.1f}%")
            with col2:
                st.metric("Uso de Memoria", f"{memory_util:.1f}%")
        
        # Detailed cluster information
        with st.expander("🔍 Información Detallada del Clúster"):
            st.json(cluster_status)
    
    else:
        st.warning(f"⚠️ Estado del clúster no disponible: {cluster_status['error']}")
        st.info("💡 Esto puede ocurrir si Ray no está completamente inicializado. Verifica los logs del contenedor.")
    
    st.stop()

# --- API STATUS ---
st.sidebar.markdown("---")
st.sidebar.subheader("Estado del Backend API")
try:
    response = requests.get('http://localhost:8000/health', timeout=2)
    if response.status_code == 200:
        st.sidebar.success("🟢 API disponible")
    else:
        st.sidebar.warning("⚠️ API con problemas")
except Exception:
    st.sidebar.error("🔴 API no disponible")


st.title("Distributed ML Platform - Visual Interface")

# --- SECTION: TRAINING ---
if section == "Training":
    st.header("🚀 Distributed Data Ingestion & Training")
    st.markdown("Enter a local directory path to scan for CSV/JSON files and upload them for distributed processing")
    
    # Initialize session state
    if 'uploaded_files' not in st.session_state:
        st.session_state['uploaded_files'] = {}
    if 'file_configs' not in st.session_state:
        st.session_state['file_configs'] = {}
    
    # Step 1: Manual Directory Input and File Discovery
    st.subheader("1. 📁 Directory Scanning and File Upload")
    
    # Manual directory path input
    st.markdown("**Enter the full path to a directory containing CSV/JSON files:**")
    
    # Examples for different operating systems
    with st.expander("💡 Directory Path Examples"):
        st.markdown("""
        **Windows Examples:**
        - `C:\\Users\\YourName\\Documents\\data`
        - `D:\\Projects\\datasets`
        
        **Linux/Mac Examples:**
        - `/home/username/data`
        - `/Users/username/Documents/datasets`
        
        **Note:** The directory should be accessible from your host machine.
        """)
    
    directory_path = st.text_input(
        "Directory Path:",
        placeholder="e.g., C:\\Users\\YourName\\Documents\\data",
        help="Enter the full path to the directory containing your CSV/JSON files"
    )
    
    # File discovery and selection
    if directory_path:
        # Normalize path separators and strip whitespace
        directory_path = directory_path.strip().replace('/', os.sep).replace('\\', os.sep)
        
        # Normalize path for comparison
        normalized_path = os.path.normpath(directory_path)
        
        if os.path.exists(normalized_path) and os.path.isdir(normalized_path):
            try:
                # Find CSV and JSON files
                all_files = os.listdir(normalized_path)
                data_files = [f for f in all_files if f.lower().endswith(('.csv', '.json'))]
                
                if data_files:
                    st.success(f"📊 Found {len(data_files)} data files in the directory")
                    
                    # Display found files
                    st.write("**Found files:**")
                    for file in data_files:
                        file_path = os.path.join(normalized_path, file)
                        try:
                            file_size = os.path.getsize(file_path)
                            size_mb = file_size / (1024 * 1024)
                            st.write(f"- {file} ({size_mb:.2f} MB)")
                        except Exception:
                            st.write(f"- {file}")
                    
                    # File selection
                    selected_files = st.multiselect(
                        "Select files to upload and process:",
                        data_files,
                        help="Choose one or more CSV/JSON files for distributed processing"
                    )
                    
                    if selected_files:
                        uploaded_count = 0
                        
                        # Upload each selected file
                        for filename in selected_files:
                            if filename not in st.session_state['uploaded_files']:
                                file_path = os.path.join(normalized_path, filename)
                                
                                try:
                                    # Read and encode file
                                    with open(file_path, 'rb') as f:
                                        file_content = f.read()
                                    encoded_content = base64.b64encode(file_content).decode('utf-8')
                                    
                                    # Upload to backend
                                    upload_request = {
                                        "filename": filename,
                                        "content": encoded_content
                                    }
                                    
                                    with st.spinner(f"Uploading {filename}..."):
                                        response = requests.post(
                                            "http://localhost:8000/upload",
                                            json=upload_request,
                                            timeout=60
                                        )
                                    
                                    if response.status_code == 200:
                                        upload_result = response.json()
                                        st.session_state['uploaded_files'][filename] = upload_result
                                        uploaded_count += 1
                                        st.success(f"✅ {filename} uploaded and distributed ({upload_result['rows']} rows)")
                                    else:
                                        st.error(f"❌ Failed to upload {filename}: {response.text}")
                                        
                                except Exception as e:
                                    st.error(f"❌ Error uploading {filename}: {e}")
                            else:
                                uploaded_count += 1
                        
                        if uploaded_count > 0:
                            st.info(f"📤 {uploaded_count} files uploaded and distributed across Ray cluster")
                            
                else:
                    st.warning("📂 No CSV or JSON files found in the specified directory")
                    
            except PermissionError:
                st.error("❌ Permission denied. Cannot access the specified directory.")
            except FileNotFoundError:
                st.error("❌ Directory not found. Please check the path.")
            except Exception as e:
                st.error(f"❌ Error accessing directory: {e}")
        else:
            st.error("❌ Directory does not exist. Please check the path and try again.")
    else:
        st.info("💡 Enter a directory path above to scan for CSV and JSON files")
    
    # Step 2: File Configuration and Training
    if st.session_state['uploaded_files']:
        st.subheader("2. ⚙️ Configure Training Parameters")
        
        # Display uploaded files and configure each one
        for filename, file_info in st.session_state['uploaded_files'].items():
            with st.expander(f"📄 Configure {filename} ({file_info['rows']} rows, {len(file_info['columns'])} columns)"):
                
                # Show file preview
                if file_info.get('preview'):
                    st.write("**Preview:**")
                    preview_df = pd.DataFrame(file_info['preview'])
                    st.dataframe(preview_df, use_container_width=True)
                
                # Task type selection
                task_type = st.selectbox(
                    "Task Type",
                    ["classification", "regression"],
                    key=f"task_{filename}"
                )
                
                # Target column selection
                target_column = st.selectbox(
                    "Target Column",
                    file_info['columns'],
                    key=f"target_{filename}"
                )
                
                # Algorithm selection
                if task_type == "classification":
                    algorithms = ["Random Forest", "XGBoost", "SVM", "Logistic Regression"]
                else:
                    algorithms = ["Random Forest Regressor", "XGBoost Regressor", "Linear Regression", "SVR"]
                
                selected_algorithm = st.selectbox(
                    "Algorithm",
                    algorithms,
                    key=f"algo_{filename}"
                )
                
                # Advanced parameters
                with st.expander("⚙️ Advanced Parameters"):
                    test_size = st.slider("Test Size", 0.1, 0.5, 0.2, key=f"test_size_{filename}")
                    random_state = st.number_input("Random State", 1, 1000, 42, key=f"random_state_{filename}")
                    cross_val_folds = st.number_input("Cross Validation Folds", 3, 10, 5, key=f"cv_folds_{filename}")
                
                # Store configuration
                st.session_state['file_configs'][filename] = {
                    'task_type': task_type,
                    'target_column': target_column,
                    'algorithm': selected_algorithm,
                    'test_size': test_size,
                    'random_state': random_state,
                    'cross_val_folds': cross_val_folds
                }
                
                # Train model button
                if st.button(f"🚀 Train Model for {filename}", key=f"train_{filename}"):
                    with st.spinner(f"Training {selected_algorithm} on {filename}..."):
                        try:
                            # Prepare training request
                            training_request = {
                                "filename": filename,
                                "task_type": task_type,
                                "target_column": target_column,
                                "algorithm": selected_algorithm.lower().replace(" ", "_"),
                                "test_size": test_size,
                                "random_state": random_state,
                                "cross_val_folds": cross_val_folds
                            }
                            
                            # Send training request
                            response = requests.post(
                                "http://localhost:8000/train",
                                json=training_request,
                                timeout=300  # 5 minutes timeout for training
                            )
                            
                            if response.status_code == 200:
                                result = response.json()
                                st.success(f"✅ Model trained successfully!")
                                st.info(f"🎯 Accuracy: {result.get('accuracy', 'N/A')}")
                                st.info(f"🧮 Model ID: {result.get('model_id', 'N/A')}")
                                
                                # Show detailed metrics if available
                                if 'metrics' in result:
                                    st.write("**Detailed Metrics:**")
                                    st.json(result['metrics'])
                            else:
                                st.error(f"❌ Training failed: {response.text}")
                                
                        except Exception as e:
                            st.error(f"❌ Error during training: {e}")

# --- SECTION: MODELS AND METRICS ---
if section == "Modelos y Métricas":
    st.header("📊 Trained Models & Performance Metrics")
    
    # Get list of trained models
    try:
        response = requests.get('http://localhost:8000/models', timeout=10)
        if response.status_code == 200:
            models = response.json()
            
            if models:
                st.success(f"📋 Found {len(models)} trained models")
                
                # Display models in expandable sections
                for model_info in models:
                    model_name = model_info.get('name', 'Unknown')
                    
                    with st.expander(f"🤖 Model: {model_name}"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Model Information:**")
                            st.write(f"- **Algorithm:** {model_info.get('algorithm', 'N/A')}")
                            st.write(f"- **Task Type:** {model_info.get('task_type', 'N/A')}")
                            st.write(f"- **Target Column:** {model_info.get('target_column', 'N/A')}")
                            st.write(f"- **Training Accuracy:** {model_info.get('accuracy', 'N/A')}")
                            
                        with col2:
                            st.write("**Actions:**")
                            
                            # Get detailed metrics
                            if st.button(f"📈 Get Metrics", key=f"metrics_{model_name}"):
                                try:
                                    metrics_response = requests.get(f'http://localhost:8000/models/{model_name}/metrics', timeout=10)
                                    if metrics_response.status_code == 200:
                                        metrics = metrics_response.json()
                                        st.write("**Detailed Metrics:**")
                                        st.json(metrics)
                                    else:
                                        st.error("Failed to get metrics")
                                except Exception as e:
                                    st.error(f"Error: {e}")
                            
                            # Visualizations
                            col_viz1, col_viz2 = st.columns(2)
                            
                            with col_viz1:
                                if st.button(f"📊 ROC Curve", key=f"roc_{model_name}"):
                                    try:
                                        roc_response = requests.get(f'http://localhost:8000/visualization/{model_name}/roc_curve', timeout=15)
                                        if roc_response.status_code == 200:
                                            st.image(roc_response.content, caption=f"ROC Curve - {model_name}")
                                        else:
                                            st.error("Failed to generate ROC curve")
                                    except Exception as e:
                                        st.error(f"Error: {e}")
                            
                            with col_viz2:
                                if st.button(f"📈 Learning Curve", key=f"learning_{model_name}"):
                                    try:
                                        learning_response = requests.get(f'http://localhost:8000/visualization/{model_name}/learning_curve', timeout=15)
                                        if learning_response.status_code == 200:
                                            st.image(learning_response.content, caption=f"Learning Curve - {model_name}")
                                        else:
                                            st.error("Failed to generate learning curve")
                                    except Exception as e:
                                        st.error(f"Error: {e}")
            else:
                st.info("📭 No trained models found. Train some models first in the Training section.")
                
        else:
            st.error(f"❌ Failed to get models: {response.text}")
            
    except Exception as e:
        st.error(f"❌ Error connecting to backend: {e}")

# --- SECTION: PREDICTION ---
if section == "Predicción":
    st.header("🔮 Model Prediction Interface")
    
    # Get list of trained models
    try:
        response = requests.get('http://localhost:8000/models', timeout=10)
        if response.status_code == 200:
            models = response.json()
            
            if models:
                # Model selection
                model_names = [model['name'] for model in models]
                selected_model = st.selectbox("Select a trained model:", model_names)
                
                if selected_model:
                    # Get model details
                    selected_model_info = next((m for m in models if m['name'] == selected_model), None)
                    
                    if selected_model_info:
                        st.write(f"**Selected Model:** {selected_model}")
                        st.write(f"**Algorithm:** {selected_model_info.get('algorithm', 'N/A')}")
                        st.write(f"**Task Type:** {selected_model_info.get('task_type', 'N/A')}")
                        
                        # Feature input method selection
                        input_method = st.radio(
                            "Choose input method:",
                            ["Manual Input", "Upload CSV File"]
                        )
                        
                        if input_method == "Manual Input":
                            st.subheader("Enter feature values manually:")
                            
                            # Get feature names from model (this would need to be stored during training)
                            st.info("💡 Enter feature values based on your trained model's expected input format")
                            
                            # Simple text area for JSON input
                            feature_input = st.text_area(
                                "Enter features as JSON:",
                                placeholder='{"feature1": 1.0, "feature2": 2.0, "feature3": "category_a"}',
                                help="Enter feature values in JSON format"
                            )
                            
                            if st.button("🔮 Make Prediction") and feature_input:
                                try:
                                    # Parse JSON input
                                    features = json.loads(feature_input)
                                    
                                    # Make prediction request
                                    prediction_request = {
                                        "model_name": selected_model,
                                        "features": features
                                    }
                                    
                                    prediction_response = requests.post(
                                        'http://localhost:8000/predict',
                                        json=prediction_request,
                                        timeout=30
                                    )
                                    
                                    if prediction_response.status_code == 200:
                                        prediction = prediction_response.json()
                                        st.success(f"✅ Prediction: {prediction.get('prediction', 'N/A')}")
                                        
                                        if 'probability' in prediction:
                                            st.info(f"🎯 Confidence: {prediction['probability']:.3f}")
                                    else:
                                        st.error(f"❌ Prediction failed: {prediction_response.text}")
                                        
                                except json.JSONDecodeError:
                                    st.error("❌ Invalid JSON format. Please check your input.")
                                except Exception as e:
                                    st.error(f"❌ Error: {e}")
                        
                        else:  # Upload CSV File
                            st.subheader("Upload CSV file for batch predictions:")
                            
                            uploaded_file = st.file_uploader(
                                "Choose a CSV file",
                                type=['csv'],
                                help="Upload a CSV file with the same feature columns as your training data"
                            )
                            
                            if uploaded_file is not None:
                                # Read and display file preview
                                df = pd.read_csv(uploaded_file)
                                st.write("**File Preview:**")
                                st.dataframe(df.head(), use_container_width=True)
                                
                                if st.button("🔮 Make Batch Predictions"):
                                    try:
                                        # Convert DataFrame to JSON
                                        data_json = df.to_json(orient='records')
                                        
                                        # Make batch prediction request
                                        batch_request = {
                                            "model_name": selected_model,
                                            "data": json.loads(data_json)
                                        }
                                        
                                        batch_response = requests.post(
                                            'http://localhost:8000/predict_batch',
                                            json=batch_request,
                                            timeout=60
                                        )
                                        
                                        if batch_response.status_code == 200:
                                            predictions = batch_response.json()
                                            
                                            # Add predictions to DataFrame
                                            df['prediction'] = predictions.get('predictions', [])
                                            if 'probabilities' in predictions:
                                                df['confidence'] = predictions['probabilities']
                                            
                                            st.success("✅ Batch predictions completed!")
                                            st.dataframe(df, use_container_width=True)
                                            
                                            # Download option
                                            csv = df.to_csv(index=False)
                                            st.download_button(
                                                label="📥 Download Results as CSV",
                                                data=csv,
                                                file_name=f"{selected_model}_predictions.csv",
                                                mime="text/csv"
                                            )
                                        else:
                                            st.error(f"❌ Batch prediction failed: {batch_response.text}")
                                            
                                    except Exception as e:
                                        st.error(f"❌ Error: {e}")
            else:
                st.info("📭 No trained models available. Train some models first in the Training section.")
                
        else:
            st.error(f"❌ Failed to get models: {response.text}")
            
    except Exception as e:
        st.error(f"❌ Error connecting to backend: {e}")
