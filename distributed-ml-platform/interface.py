import streamlit as st
import requests
import json
import os
import pandas as pd

st.set_page_config(page_title="Distributed ML Platform Interface", layout="wide")

# --- CLUSTER MANAGEMENT ---
st.sidebar.title("Menú principal")
section = st.sidebar.radio("Selecciona una sección", ["Cluster", "Training", "Modelos y Métricas", "Predicción"])

if 'cluster' not in st.session_state:
    st.session_state['cluster'] = {
        'head': {'cpu': 2, 'ram': 4, 'running': False},
        'workers': [],  # List of dicts: {'cpu': X, 'ram': Y, 'running': False}
        'max_workers': 4
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

def start_worker(idx, cpu, ram):
    """Workers in containerized setup would be managed by scaling the backend service"""
    try:
        response = requests.post('http://localhost:8000/cluster/add_worker', json={'cpu': cpu, 'ram': ram}, timeout=10)
        if response.status_code == 200:
            st.session_state['cluster']['workers'][idx]['running'] = True
            return True
        st.error(f"Error en backend: {response.text}")
        return False
    except Exception as e:
        st.error(f"Error conectando con backend: {e}")
        return False

def stop_worker(idx):
    """Workers in containerized setup would be managed by scaling the backend service"""
    try:
        response = requests.post('http://localhost:8000/cluster/remove_worker', json={'worker_id': idx}, timeout=10)
        if response.status_code == 200:
            st.session_state['cluster']['workers'][idx]['running'] = False
            return True
        st.error(f"Error en backend: {response.text}")
        return False
    except Exception as e:
        st.error(f"Error conectando con backend: {e}")
        return False

if section == "Cluster":
    st.header("Gestión del Clúster Ray Distribuido")
    
    # Get real cluster status from backend
    cluster_status = get_cluster_status()
    
    if "error" not in cluster_status:
        st.subheader("Estado Actual del Clúster")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Nodos Activos", cluster_status.get("nodes", 0))
        with col2:
            st.metric("CPUs Totales", cluster_status.get("cluster_resources", {}).get("CPU", 0))
        with col3:
            st.metric("CPUs Disponibles", cluster_status.get("available_resources", {}).get("CPU", 0))
        
        with st.expander("Detalles del Clúster"):
            st.json(cluster_status)
    else:
        st.error(f"No se pudo obtener el estado del clúster: {cluster_status['error']}")
    
    cluster = st.session_state['cluster']

    st.subheader("Head Node (Gestionado por Docker)")
    col1, col2, col3 = st.columns([1,1,2])
    with col1:
        head_cpu = st.number_input("Head CPU", min_value=1, max_value=16, value=cluster['head']['cpu'], key="head_cpu")
    with col2:
        head_ram = st.number_input("Head RAM (GB)", min_value=1, max_value=32, value=cluster['head']['ram'], key="head_ram")
    with col3:
        st.info("El head node se gestiona automáticamente con Docker Compose")
    st.write("Estado: 🟢 Activo (gestionado por Docker)")

    st.subheader("Workers")
    st.info("En el setup actual, los workers se gestionan escalando el servicio backend o creando contenedores adicionales.")
    
    for idx, worker in enumerate(cluster['workers']):
        col1, col2, col3, col4 = st.columns([1,1,1,2])
        with col1:
            cpu = st.number_input(f"CPU Worker {idx+1}", min_value=1, max_value=16, value=worker['cpu'], key=f"worker_cpu_{idx}")
        with col2:
            ram = st.number_input(f"RAM Worker {idx+1} (GB)", min_value=1, max_value=32, value=worker['ram'], key=f"worker_ram_{idx}")
        with col3:
            st.write(f"{'🟢' if worker['running'] else '🔴'}")
        with col4:
            if not worker['running']:
                if st.button(f"Solicitar Worker {idx+1}", key=f"start_worker_{idx}"):
                    if start_worker(idx, cpu, ram):
                        worker['cpu'] = cpu
                        worker['ram'] = ram
                        st.success(f"Worker {idx+1} solicitado")
                        st.experimental_rerun()
            else:
                if st.button(f"Detener Worker {idx+1}", key=f"stop_worker_{idx}"):
                    if stop_worker(idx):
                        st.success(f"Worker {idx+1} detenido")
                        st.experimental_rerun()
    
    if len(cluster['workers']) < cluster['max_workers']:
        if st.button("Agregar Worker"):
            cluster['workers'].append({'cpu': 2, 'ram': 4, 'running': False})
            st.experimental_rerun()
    if cluster['workers']:
        if st.button("Eliminar Último Worker"):
            cluster['workers'].pop()
            st.experimental_rerun()
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
    st.header("1. Selección de Directorio de Datasets")
    data_dir = st.text_input("Ruta del directorio de datasets (dentro del contenedor Streamlit)")
    
    if data_dir and os.path.isdir(data_dir):
        files = [f for f in os.listdir(data_dir) if f.endswith('.csv') or f.endswith('.json')]
        if files:
            st.success(f"{len(files)} archivos encontrados.")
            selected_files = st.multiselect("Selecciona uno o más datasets", files)
            dataset_columns = {}
            for fname in selected_files:
                fpath = os.path.join(data_dir, fname)
                try:
                    if fname.endswith('.csv'):
                        df = pd.read_csv(fpath, nrows=100)
                    else:
                        df = pd.read_json(fpath, nrows=100)
                    st.write(f"**{fname}**: {df.shape[0]} filas, {df.shape[1]} columnas")
                    col = st.selectbox(f"Columna objetivo para {fname}", df.columns, key=f"target_{fname}")
                    dataset_columns[fname] = col
                except Exception as e:
                    st.warning(f"Error leyendo {fname}: {e}")
            if selected_files and st.button("Listo para entrenar"):
                st.session_state['selected_datasets'] = selected_files
                st.session_state['target_columns'] = dataset_columns
                st.success("Datasets y columnas objetivo seleccionados. Continúa con la configuración de modelos.")
                st.stop()
        else:
            st.warning("No se encontraron archivos CSV o JSON en el directorio.")
    elif data_dir:
        st.error("Directorio no válido o no existe.")
    else:
        st.info("Introduce la ruta de un directorio con datasets.")

    # Only show model selection if datasets are selected
    if 'selected_datasets' in st.session_state and 'target_columns' in st.session_state:
        st.header("2. Configuración de Modelos y Entrenamiento Distribuido")
        selected_files = st.session_state['selected_datasets']
        dataset_columns = st.session_state['target_columns']
        task_types = {}
        model_choices = {}
        for fname in selected_files:
            st.subheader(f"{fname}")
            task = st.radio(f"Tipo de tarea para {fname}", ["Clasificación", "Regresión"], key=f"task_{fname}")
            task_types[fname] = task
            if task == "Clasificación":
                models = ["RandomForestClassifier", "LogisticRegression", "SVC", "GradientBoostingClassifier", "KNeighborsClassifier"]
            else:
                models = ["RandomForestRegressor", "LinearRegression", "Ridge", "Lasso", "GradientBoostingRegressor", "ElasticNet"]
            selected = st.multiselect(f"Modelos para {fname}", models, key=f"models_{fname}")
            model_choices[fname] = selected
        if st.button("Entrenar Modelos Distribuidos"):
            # Prepare training request
            datasets = {}
            for fname in selected_files:
                fpath = os.path.join(data_dir, fname)
                if fname.endswith('.csv'):
                    df = pd.read_csv(fpath)
                else:
                    df = pd.read_json(fpath)
                datasets[fname] = df.to_dict('records')
            
            training_request = {
                'datasets': datasets,
                'ml_tasks': task_types,
                'targets': dataset_columns,
                'model_selections': model_choices  # Include model selections
            }
            
            with st.spinner("Iniciando entrenamiento distribuido..."):
                try:
                    response = requests.post("http://localhost:8000/train", json=training_request, timeout=300)
                    if response.status_code == 200:
                        result = response.json()
                        st.success("✅ Entrenamiento distribuido completado!")
                        st.write("**Resultados del entrenamiento:**")
                        for dataset_name, dataset_result in result['results'].items():
                            st.write(f"**{dataset_name}:**")
                            st.write(f"- Modelos entrenados: {dataset_result['models_trained']}")
                            st.write(f"- Modelos: {', '.join(dataset_result['model_names'])}")
                            with st.expander(f"Ver métricas de {dataset_name}"):
                                st.json(dataset_result['metrics'])
                        # Reset session state for next run
                        st.session_state.pop('selected_datasets', None)
                        st.session_state.pop('target_columns', None)
                    else:
                        st.error(f"Error en entrenamiento: {response.text}")
                except requests.exceptions.Timeout:
                    st.error("El entrenamiento está tardando más de lo esperado. Verifica el estado en la sección de Modelos y Métricas.")
                except Exception as e:
                    st.error(f"Error conectando con el backend: {e}")

# --- SECTION: MODELS & METRICS ---
elif section == "Modelos y Métricas":
    st.header("Modelos Entrenados y Métricas")
    try:
        resp = requests.get("http://localhost:8000/models")
        if resp.status_code == 200:
            models = resp.json().get("models", [])
            if models:
                selected_models = st.multiselect("Selecciona modelos para ver métricas", models)
                if selected_models:
                    col1, col2 = st.columns(2)
                    with col1:
                        show_metrics = st.button("Ver Métricas")
                    with col2:
                        show_visualizations = st.button("Ver Visualizaciones")
                    
                    if show_metrics:
                        for model_name in selected_models:
                            st.subheader(f"Métricas: {model_name}")
                            mresp = requests.get(f"http://localhost:8000/metrics/{model_name}")
                            if mresp.status_code == 200:
                                st.json(mresp.json())
                            else:
                                st.warning(f"No se pudieron obtener métricas para {model_name}")
                    
                    if show_visualizations:
                        for model_name in selected_models:
                            st.subheader(f"Visualizaciones: {model_name}")
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write("**Curva ROC**")
                                try:
                                    roc_url = f"http://localhost:8000/visualization/{model_name}/roc"
                                    st.image(roc_url, caption=f"ROC - {model_name}")
                                except Exception as e:
                                    st.error(f"Error cargando ROC: {e}")
                            
                            with col2:
                                st.write("**Curva de Aprendizaje**")
                                try:
                                    learning_url = f"http://localhost:8000/visualization/{model_name}/learning_curve"
                                    st.image(learning_url, caption=f"Learning Curve - {model_name}")
                                except Exception as e:
                                    st.error(f"Error cargando curva de aprendizaje: {e}")
                            
                            st.markdown("---")
                
                # Add dashboard link
                st.subheader("Dashboard Completo")
                if st.button("Ver Dashboard de Todos los Modelos"):
                    dashboard_url = "http://localhost:8000/visualization/all"
                    st.markdown(f"[🎯 Abrir Dashboard Completo]({dashboard_url})")
                    st.info("El dashboard se abrirá en una nueva pestaña con todas las visualizaciones.")
            else:
                st.info("No hay modelos entrenados disponibles.")
        else:
            st.error("No se pudo obtener la lista de modelos.")
    except Exception as e:
        st.error(f"Error consultando modelos: {e}")

# --- SECTION: PREDICTION ---
elif section == "Predicción":
    st.header("Realizar Predicción con un Modelo Entrenado")
    try:
        resp = requests.get("http://localhost:8000/models")
        if resp.status_code == 200:
            models = resp.json().get("models", [])
            if models:
                selected_model = st.selectbox("Selecciona un modelo para predecir", models)
                if selected_model:
                    st.info("Introduce las características en formato JSON (una muestra o lista de muestras):")
                    input_features = st.text_area("Características", "{\n  \"feature1\": 1.0,\n  \"feature2\": 2.0\n}", height=150)
                    if st.button("Realizar Predicción"):
                        try:
                            features = json.loads(input_features.replace("'", '"'))
                            features_list = [features] if isinstance(features, dict) else features
                            pred_resp = requests.post(f"http://localhost:8000/predict/{selected_model}", json={"features": features_list})
                            if pred_resp.status_code == 200:
                                st.success("✅ Predicción realizada:")
                                st.json(pred_resp.json())
                            else:
                                st.error(f"Error en predicción: {pred_resp.text}")
                        except Exception as e:
                            st.error(f"Error en predicción: {e}")
            else:
                st.info("No hay modelos entrenados disponibles.")
        else:
            st.error("No se pudo obtener la lista de modelos.")
    except Exception as e:
        st.error(f"Error consultando modelos: {e}")
        st.warning("� Selecciona un directorio válido con datasets o usa los datos de ejemplo.")
