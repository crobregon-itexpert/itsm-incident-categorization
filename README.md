# IT service desk incident categorization (ML, transformers) and a first-contact LLM agent

Code, configuration, run logs and an anonymized dataset for the paper **"IT Service Desk Automation Using ML
and LLM"** (C. Obregón; revised version of manuscript AIEng-26010199).

- **Phase 1.** A static taxonomy of 10 incident categories is derived from 59,860 real service-desk incidents
  (Spanish, restaurant sector) and six TF-IDF classifiers and two fine-tuned transformers (BETO, XLM-RoBERTa)
  are compared on five stratified splits, with a view on titles never seen in training.
- **Phase 2.** The classifier is deployed behind a first-contact agent (Vertex AI, Gemini) in Microsoft Teams;
  its retrieval step and the agent itself are evaluated.

## Repository layout
| Path | Content |
|---|---|
| `data/itsm_incidents_2024_public.csv` | Anonymized dataset (59,860 incidents, 10 categories, 5 splits). See `docs/dataset_card.md` |
| `configs/` | Experiment configuration (`phase1_v2.yaml`) and category names/definitions (`categories.yaml`) |
| `src/itsm/` | Pipeline: cleaning, anonymization, clustering, splits, training, evaluation, reports |
| `notebooks/` | Colab notebooks for the transformer search and runs (GPU) |
| `results/runs/phase1_v2/` | Every run: configuration, environment, metrics, confusion matrix, test predictions (by `tid`) |
| `results/summary/` | Aggregated results used in the paper's tables |
| `paper/` | LaTeX source, generated tables and figures, PDF |
| `agent/` | Deployed components: classifier/retrieval service (`classifier/`), agent flow and automated evaluation (`backend/src/`), Gemini baseline script |

## What can be reproduced from the public data
| Step | Command | Data needed |
|---|---|---|
| Classical models (Table II) | `PYTHONPATH=src python -m itsm.train_classical --config configs/phase1_v2.yaml` | public CSV |
| Transformers (Table II, X) | `notebooks/colab_transformer_search_phase1.ipynb` (GPU) | public CSV |
| Tables and figures | `PYTHONPATH=src python -m itsm.report --config configs/phase1_v2.yaml` | run logs in `results/` |
| Confidence analysis (Table VI) | `PYTHONPATH=src python -m itsm.confidence` | run logs |
| Cleaning, anonymization, clustering, retrieval and agent evaluation | `itsm.prepare`, `itsm.clustering`, `itsm.retrieval_eval`, ... | private source data (not released) |

The public file differs from the file used in the experiments only by a one-to-one substitution of
organization identifiers (see the dataset card): word-level models reproduce exactly (e.g. logistic regression,
seed 0: 83.4% accuracy, as logged); character n-gram and transformer models may differ slightly.

Install: `pip install -r requirements.txt` (Python 3.11+; transformers and torch only for the notebooks).

## Agent (Phase 2)
`agent/classifier/app.py` is the private Cloud Run service (MLP over TF-IDF, confidence threshold 0.8, TF-IDF
retrieval over anonymized resolved incidents); its model and knowledge base are built by
`src/itsm/agent_assets.py` from the private data and are not released. `agent/backend/src/agent.js` implements
the Teams flow (intake, classification and retrieval, grounded self-help steps, confirmation, ticket), and
`eval.js` the automated evaluation (simulated user + LLM judge). Credentials are read from environment
variables / Secret Manager; none are included.

## License
Code: MIT. Dataset: CC BY 4.0. See `LICENSE`.

## Citation
If you use this code or data, please cite the paper (reference to be completed on publication).
