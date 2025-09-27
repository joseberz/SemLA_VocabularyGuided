PROJECT_NAME=semla

.PHONY: venv sync clean

venv:
	uv venv --seed --python 3.10

# Install Detectron2 after syncing to avoid build isolation issues
install: venv 
	uv sync
	uv pip install --no-build-isolation https://github.com/facebookresearch/detectron2.git

clean:
	rm -rf .venv uv.lock