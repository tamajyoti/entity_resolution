.ONESHELL:
SHELL := /bin/bash
SRC = $(wildcard ./*.ipynb)
DIR := am_combiner
TEST_DIR := test
MAX_LINE_LENGTH := 100
FORMAT_EXCLUSIONS := "_nbdev|ab_utils|articles|config|date|entity_linking|profession|name_commonness"
STYLE_EXCLUSIONS := _nbdev.py,ab_utils.py,articles.py,config.py,date.py,entity_linking.py,profession.py,name_commonness.py
DOCSTYLE_INCLUSIONS := '^(?!_nbdev)(?!ab_utils)(?!articles)(?!config)(?!date)(?!entity_linking)(?!profession)(?!name_commonness).*\.py'
STYLE_IGNORED_ERROR_CODES := E203,W605
DOCSTYLE_IGNORED_ERROR_CODES := D100,D104,D107,D211,D212

all: am_combiner docs

start_test_runs:
	helm -n entity-resolution install make-tfidf-cosine-similarity helm/ -f helm/values.yaml -f helm/tfidf-cosine-similarity.yaml
	helm -n entity-resolution install make-tfidf-and-spectral-clustering helm/ -f helm/values.yaml -f helm/tfidf-and-spectral-clustering.yaml
	helm -n entity-resolution install make-tfidf-and-features helm/ -f helm/values.yaml -f helm/tfidf-and-features.yaml
	helm -n entity-resolution install make-gcn-combiner helm/ -f helm/values.yaml -f helm/gcn-combiner.yaml

stop_test_runs:
	helm uninstall make-tfidf-cosine-similarity
	helm uninstall make-tfidf-and-spectral-clustering
	helm uninstall make-tfidf-and-features
	helm uninstall make-gcn-combiner

start_best_models_run:
	helm -n entity-resolution install tfidf-cosine-best-model-set-a helm/ -f helm/values.yaml -f helm/tfidf-cosine-similarity-best-model-set-a.yaml
	helm -n entity-resolution install tfidf-cosine-best-model-set-b helm/ -f helm/values.yaml -f helm/tfidf-cosine-similarity-best-model-set-b.yaml
	helm -n entity-resolution install tfidf-cosine-best-model-set-c helm/ -f helm/values.yaml -f helm/tfidf-cosine-similarity-best-model-set-c.yaml

	helm -n entity-resolution install tfidf-features-set-a helm/ -f helm/values.yaml -f helm/tfidf-and-features-best-model-set-a.yaml
	helm -n entity-resolution install tfidf-features-set-b helm/ -f helm/values.yaml -f helm/tfidf-and-features-best-model-set-b.yaml
	helm -n entity-resolution install tfidf-features-set-c helm/ -f helm/values.yaml -f helm/tfidf-and-features-best-model-set-c.yaml

stop_best_models_run:
	helm uninstall tfidf-cosine-best-model-set-a
	helm uninstall tfidf-cosine-best-model-set-b
	helm uninstall tfidf-cosine-best-model-set-c

	helm uninstall tfidf-features-set-a
	helm uninstall tfidf-features-set-b
	helm uninstall tfidf-features-set-c

am_combiner: $(SRC)
	nbdev_build_lib
	touch am_combiner

sync:
	nbdev_update_lib

docs_serve: docs
	cd docs && bundle exec jekyll serve

docs: $(SRC)
	nbdev_build_docs
	touch docs

test:
	nbdev_test_nbs

cov_integtest:
	coverage run --source $(DIR) -m pytest -m integtest --junitxml=tests.xml --cov-report term-missing --cov-report xml

cov_test:
	coverage run --source $(DIR) -m pytest -m "not integtest" --junitxml=tests.xml --cov-report term-missing --cov-report xml

cov_report:
	coverage report -m

cov_artefacts:
	coverage xml -o coverage.xml

format:
	black $(DIR) $(TEST_DIR) -l $(MAX_LINE_LENGTH) --exclude $(FORMAT_EXCLUSIONS) --check

style:
	flake8 $(DIR) $(TEST_DIR) --max-line-length $(MAX_LINE_LENGTH) --exclude $(STYLE_EXCLUSIONS) --extend-ignore $(STYLE_IGNORED_ERROR_CODES)

docstyle:
	pydocstyle $(DIR) --match $(DOCSTYLE_INCLUSIONS) --ignore $(DOCSTYLE_IGNORED_ERROR_CODES)

release: pypi
	nbdev_conda_package
	nbdev_bump_version

pypi: dist
	twine upload --repository pypi dist/*

dist: clean
	python setup.py sdist bdist_wheel

clean:
	rm -rf dist
