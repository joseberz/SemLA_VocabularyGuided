PROJECT_NAME=semla

.PHONY: init add-build-dependencies sync clean

venv:
	uv venv --seed --python 3.10

# Install without build isolation so that Detectron can use torch installed in previous step
sync: venv
	uv sync

clean:
	rm -rf .venv uv.lock