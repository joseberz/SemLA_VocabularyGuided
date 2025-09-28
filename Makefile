PROJECT_NAME=semla

.PHONY: install clean

# Install Detectron2 after syncing to avoid build isolation issues
install:
	unset PYTHONPATH
	uv venv --seed --python 3.10
	uv sync
	uv pip install --no-build-isolation https://github.com/facebookresearch/detectron2.git

clean:
	rm -rf .venv uv.lock