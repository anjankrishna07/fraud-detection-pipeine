.PHONY: help setup ingest validate features train serve monitor test clean ui

help:
	@echo "Fraud Detection ML Platform - Makefile Commands"
	@echo "================================================"
	@echo "  make setup       - Install dependencies"
	@echo "  make ingest      - Download/load raw data"
	@echo "  make validate    - Validate raw data"
	@echo "  make features     - Build features from raw data"
	@echo "  make train        - Train model"
	@echo "  make serve        - Start API server"
	@echo "  make ui           - Start Streamlit web interface"
	@echo "  make monitor      - Run monitoring pipeline"
	@echo "  make test         - Run smoke tests"
	@echo "  make clean        - Clean generated files"
	@echo ""

setup:
	pip install --upgrade pip
	pip install -e .
	@echo "✓ Dependencies installed"

ingest:
	python -m fraud_platform.ingestion.load_raw --use-kaggle-api

validate:
	python -m fraud_platform.validation.run_validation

features:
	python -m fraud_platform.features.build_features

train:
	python -m fraud_platform.training.train

serve:
	python -m fraud_platform.serving.app

ui:
	streamlit run streamlit_app.py

monitor:
	python -m fraud_platform.pipelines.run_monitoring_pipeline

test:
	pytest tests/ -v

clean:
	rm -rf data/processed/*
	rm -rf data/features/*
	rm -rf reports/*
	rm -rf __pycache__/
	rm -rf .pytest_cache/
	find . -type d -name __pycache__ -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	@echo "✓ Cleaned generated files"

docker-build:
	docker-compose build

docker-train:
	docker-compose run --rm training

docker-serve:
	docker-compose up api

docker-monitor:
	docker-compose run --rm monitoring
