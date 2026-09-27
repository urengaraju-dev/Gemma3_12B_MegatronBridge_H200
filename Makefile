# Convenience targets for the Gemma-3-12B LoRA pipeline.
# Usage: make <target>

CONTAINER ?= mbridge
SHELL := /bin/bash

.PHONY: help setup smoke shell logs clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

setup: ## Provision container + weights (scripts/setup.sh)
	bash scripts/setup.sh

smoke: ## Run the LoRA smoke test (scripts/run_smoke.sh)
	bash scripts/run_smoke.sh

shell: ## Open a shell inside the running container
	docker exec -it $(CONTAINER) bash

logs: ## Tail the most recent training log inside the container
	docker exec $(CONTAINER) bash -lc 'tail -f "$$(ls -t /workspace/logs/*.out | head -1)"'

clean: ## Remove the container and local run artifacts
	-docker rm -f $(CONTAINER)
	rm -rf logs outputs nemo_experiments
