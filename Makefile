PROJECT_NAME=semla

.PHONY: venv sync clean

venv:
	uv venv --seed --python 3.10

# Install without build isolation so that Detectron can use torch installed in previous step
sync: venv 
	uv sync
	# Add detectron after installing Torch because it is required
	uv pip install --no-build-isolation https://github.com/facebookresearch/detectron2.git

clean:
	rm -rf .venv uv.lock