# Airflow image: official Airflow base + the project ML stack + DAGs/code.
# DAGs import `src.*`, so the package ships inside the image (mounted volumes
# override it in dev for hot-reload).
FROM apache/airflow:2.10.4-python3.11

ENV PYTHONPATH=/opt/airflow

COPY requirements-airflow.txt /tmp/requirements-airflow.txt
RUN pip install --no-cache-dir -r /tmp/requirements-airflow.txt

USER root
RUN mkdir -p /opt/airflow/data /opt/airflow/models && chown -R airflow:root /opt/airflow
USER airflow

COPY --chown=airflow:root src /opt/airflow/src
COPY --chown=airflow:root config /opt/airflow/config
COPY --chown=airflow:root airflow/dags /opt/airflow/dags
