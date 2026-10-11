# Shortcuts - run from the project root (needs Docker; `make` on Windows via Git Bash/WSL)
.PHONY: build up pipeline test eda clean

build:        ## build the Docker image
	docker-compose build

up:           ## start JupyterLab at http://localhost:8888
	docker-compose up

pipeline:     ## run the full bronze -> silver -> gold backfill
	docker-compose run --rm jupyter python main.py

test:         ## run unit tests for the cleaning rules and validation
	docker-compose run --rm jupyter python -m pytest tests -q

eda:          ## re-execute the exploration notebook
	docker-compose run --rm -w /app/notebooks jupyter jupyter nbconvert --to notebook --execute --inplace 01_eda.ipynb

clean:        ## delete the generated datamart
	rm -rf datamart
