FROM python:3.9-slim

# Establecer directorio de trabajo
WORKDIR /app

# Copiar requirements primero para mejor caché
COPY requirements.txt .

# Instalar todas las dependencias sin verificar hashes (usando --hash=skip)
RUN pip install --no-cache-dir --hash=skip -r requirements.txt

# Copiar archivos del proyecto
COPY . .

# Exponer puerto para API
EXPOSE 8000

# Comando de inicio con argumento predeterminado
ENTRYPOINT ["python", "src/main.py"]
CMD ["--mode=serve", "--config=src/config/config.yaml"]
