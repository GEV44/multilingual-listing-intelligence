# Changelog

All notable changes to this project are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow SemVer.

## [0.1.0] - 2026-11-13

### Added
- Typed schema (`Listing`, `BuyerRequest`) and `LI_`-prefixed typed settings.
- Rule-based RU/HY/EN extractor: deal type, price and currency, rooms, area, floor of total,
  district, street, renovation, furniture, pets, heating.
- `HybridExtractor` with a pluggable `ModelBackend`; model output is schema-validated and only
  fills fields the rules left empty. Optional `hf` extra for a local Hugging Face model.
- Synthetic data generator: labelled posts (standard and hard tiers), buyer requests, graded
  relevance, procedural photo sets with re-post perturbations.
- De-duplication: 64-bit DCT perceptual hash, fuzzy spec similarity, union-find clustering.
- Matching: hard filters, explainable weighted scoring, budget widening, learned weights.
- Evaluation harness (field-level P/R/F1, nDCG@k, pairwise dedup F1) with a results table.
- FastAPI service (`/health`, `/extract`, `/match`, `/dedup`), Dockerfile, docker-compose.
- Ruff, mypy (strict), pytest with coverage gate, pre-commit, GitHub Actions CI.
