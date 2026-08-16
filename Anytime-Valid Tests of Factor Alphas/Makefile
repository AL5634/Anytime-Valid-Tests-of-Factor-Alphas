# E-value Factor Zoo — Makefile
# Target journal: Journal of Empirical Finance

.PHONY: help data evalue oos robust figures analysis all clean test

help:
	@echo "Targets:"
	@echo "  make data      - build the factor panel (CZ 212 or Ken French)"
	@echo "  make evalue    - compute e-values for all factors"
	@echo "  make oos       - run OOS horse race (e-value vs HLZ/EB-shrink/t>2)"
	@echo "  make robust    - power simulation + min_obs x clip sensitivity grid"
	@echo "  make figures   - generate figures"
	@echo "  make analysis  - evalue + oos + robust + figures"
	@echo "  make all       - data + analysis"
	@echo "  make test      - run unit tests"
	@echo "  make clean     - remove generated artifacts"

data:
	python -m src.run --stage data

evalue:
	python -m src.run --stage evalue

oos:
	python -m src.run --stage oos

robust:
	python -m src.run --stage robust

figures:
	python -m src.run --stage figures

analysis: evalue oos robust figures

all: data analysis

test:
	python -m pytest tests -v

clean:
	rm -rf data/processed/* data/results/* output/tables/* output/figures/*
