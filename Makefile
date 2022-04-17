REPO_NAME := $(shell basename `git rev-parse --show-toplevel`)
DVC_REMOTE := ${GDRIVE_FOLDER}/${REPO_NAME}


.PHONY:test
test:
	poetry run python -m pytest

.PHONY:install-hooks
install-hooks:
	precommit install

.PHONY:tensorboard
tensorboard:
	poetry run tensorboard --logdir=runs

.PHONY:dvc
dvc:
	dvc init
	dvc remote add --default gdrive ${DVC_REMOTE}

.PHONY: mlflow
mlflow:
	poetry run mlflow ui

.PHONY: prefect
prefect:
	poetry run prefect server start --use-volume
