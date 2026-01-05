# 
# -------------------------------------------------------------------
# Shell & platform
# -------------------------------------------------------------------
SHELL := /usr/bin/bash

# -------------------------------------------------------------------
# Python / venv configuration
# -------------------------------------------------------------------
VENV := .venv

ifeq ($(OS),Windows_NT)
	PYTHON := $(VENV)/Scripts/python.exe
	PIP := $(VENV)/Scripts/pip.exe
	AGENTIC := $(VENV)/Scripts/agentic-rag.exe
	PYTEST := $(VENV)/Scripts/pytest.exe
else
	PYTHON := $(VENV)/bin/python
	PIP := $(VENV)/bin/pip
	AGENTIC := $(VENV)/bin/agentic-rag
	PYTEST := $(VENV)/bin/pytest
endif

# -------------------------------------------------------------------
# Project paths
# -------------------------------------------------------------------
DATA_DIR ?= data/raw
PROCESSED_DIR ?= data/processed

# -------------------------------------------------------------------
# Phony targets
# -------------------------------------------------------------------
.PHONY: help venv install data ingest agent eval test clean compose-up compose-down

# -------------------------------------------------------------------
# Help
# -------------------------------------------------------------------
help:
	@echo "Available targets:"
	@echo "  make venv         Create virtualenv"
	@echo "  make install      Install dependencies"
	@echo "  make data         Download dataset into $(DATA_DIR)"
	@echo "  make ingest       Run ingestion pipeline"
	@echo "  make agent        Launch agent controller"
	@echo "  make eval         Run retrieval/agent evaluations"
	@echo "  make test         Run pytest"
	@echo "  make compose-up   Start pgvector stack"
	@echo "  make compose-down Stop pgvector stack"
	@echo "  make clean        Remove venv and generated data"

# -------------------------------------------------------------------
# Virtual environment
# -------------------------------------------------------------------
venv:
	@if [ ! -d "$(VENV)" ]; then python -m venv "$(VENV)"; fi

# -------------------------------------------------------------------
# Install dependencies
# -------------------------------------------------------------------
install: venv
	$(PYTHON) -m pip install -U pip
	$(PIP) install -r requirements.txt

# -------------------------------------------------------------------
# Data / pipelines
# -------------------------------------------------------------------
data: install
	$(PYTHON) scripts/download_dataset.py --output "$(DATA_DIR)"

ingest: install
	$(AGENTIC) ingest --raw-dir "$(DATA_DIR)" --output-dir "$(PROCESSED_DIR)"

agent: install
	$(AGENTIC) agent

eval: install
	$(AGENTIC) evaluate

test: install
	$(PYTEST) -q

# -------------------------------------------------------------------
# Docker
# -------------------------------------------------------------------
compose-up:
	docker compose up -d vectorstore

compose-down:
	docker compose down

# -------------------------------------------------------------------
# Cleanup
# -------------------------------------------------------------------
clean:
	rm -rf "$(VENV)" "$(DATA_DIR)" "$(PROCESSED_DIR)"
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
