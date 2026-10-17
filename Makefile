# Convenience targets. Run from the repository root inside your virtualenv.
IMAGE ?= heart-risk-api
TAG   ?= latest
N_ITER ?= 30

.PHONY: install data eda train mlflow-ui lint test serve docker-build docker-run smoke \
        compose-up compose-down k8s-deploy k8s-delete all

install:            ## clean install of everything
	pip install -r requirements-dev.txt

data:               ## download + clean the UCI data -> data/processed/heart_clean.csv
	python -m heart.data

eda:                ## EDA figures -> reports/figures/eda
	python -m heart.eda

train:              ## tune, track in MLflow, save models/model.joblib
	python -m heart.train --n-iter $(N_ITER)

mlflow-ui:
	mlflow ui --backend-store-uri ./mlruns --port 5000

lint:
	ruff check .

test:
	pytest --cov=heart --cov=api --cov-report=term-missing

serve:
	uvicorn api.main:app --reload --port 8000

docker-build:
	docker build -t $(IMAGE):$(TAG) .

docker-run:
	docker run --rm -p 8000:8000 --name heart-api $(IMAGE):$(TAG)

smoke:              ## hit a running API with the sample patient
	bash scripts/smoke_test.sh http://localhost:8000

compose-up:         ## API + Prometheus + Grafana locally
	docker compose up --build -d

compose-down:
	docker compose down

k8s-deploy:
	bash scripts/k8s_deploy.sh

k8s-delete:
	kubectl delete namespace heart-risk --ignore-not-found

all: data eda train lint test docker-build
