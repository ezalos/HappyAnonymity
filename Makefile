# ABOUTME: Convenience targets for building, running, and comparing face anonymization
# ABOUTME: Wraps docker compose commands for common operations

INPUT ?= data/input/walking_nyc.webm
TRACKER ?= sort

.PHONY: build run process compare detect download-models

build:
	docker compose build

download-models:
	uv run python scripts/download_models.py --model-dir models

process:
	docker compose run --rm anonymize process \
		--input $(INPUT) \
		--output data/output/$$(basename $(INPUT) .webm)_$(TRACKER).mp4 \
		--tracker $(TRACKER)

compare:
	docker compose run --rm anonymize compare \
		--input $(INPUT) \
		--output-dir data/output/comparison/

detect:
	docker compose run --rm anonymize detect \
		--input $(INPUT) \
		--output data/output/$$(basename $(INPUT) .webm)_detections.mp4

# Run locally without Docker
run:
	uv run python -m src.cli $(ARGS)
