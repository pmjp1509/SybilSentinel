<h1 align="center">SybilSentinel</h1>

## Sybil Attacks on Federated Learning — attack, similarity-filter defense, comparison and web app

Extension of [ITSEG-MQ/Sybil-Attacks-and-Defense-on-DP-FL](https://github.com/ITSEG-MQ/Sybil-Attacks-and-Defense-on-DP-FL)
(FedAvg + Laplace "DP" noise on MNIST/CIFAR-10). It adds:

* **a coordinated Sybil attack** (one adversary, many fake clients): label-flip, backdoor, sign-flip, random, the original DP-noise attack, plus an adaptive "evasive" setting
* **our novelty: a cosine-similarity Sybil filter on top of the DP noise** (FoolsGold-style), and a **hybrid** that follows it with a coordinate-wise median
* **comparison defenses** from the literature: Multi-Krum, median, trimmed mean, FoolsGold-style weighting
* **benchmarks + plots** (accuracy, attack success, detection TPR/FPR) and a **Streamlit web interface**

## 1. Setup and run

```bash
# Python 3.9+ ; CPU is enough
python -m venv venv
venv\Scripts\activate               # linux: source venv/bin/activate
pip install -r requirements.txt     # torch, numpy, matplotlib, streamlit, pandas

python tests/test_filter.py         # unit tests of the similarity filter
python tests/test_aggregators.py    # unit tests of the comparison defenses

# --- the web interface (recommended for demos) ---
streamlit run app.py                # opens http://localhost:8501

# --- single runs from the command line (MNIST downloads into ./data on first run) ---
python main_fed_avg_sybil.py --sybil 0                         # baseline, no attack
python main_fed_avg_sybil.py --sybil 20                        # 20% Sybils, no defense
python main_fed_avg_sybil.py --sybil 20 --enable_filter        # + our similarity filter
python main_fed_avg_sybil.py --sybil 30 --attack backdoor --defense filter_median

# --- benchmarks ---
python benchmark.py                 # 0/20/40% Sybils x (no defense | filter), 50 rounds, ~12 min on 2 CPU cores
python benchmark_compare.py         # 7 defenses x 3 attacks at 30% Sybils, 30 rounds, ~30 min on 2 CPU cores
python benchmark.py --epochs 20     # quick version (same flag exists on benchmark_compare.py)
```

Outputs: `results/` (scale benchmark: JSON per run, `summary.md`, `acc_vs_rounds.png`, `acc_vs_sybil.png`, `detection.png`)
and `results/compare/` (`compare_accuracy.png`, `compare_attack_success.png`, `compare_summary.md`). Runs started from the web
interface are saved in `results/ui/`. Both benchmarks skip runs whose JSON already exists (delete the folder to redo).
Re-plot without re-training: `python plot_results.py`, `python plot_compare.py`.
If the MNIST download is blocked, put the four `*-ubyte.gz` files into `data/mnist/raw/`. GPU: add `--gpu 0` to a single run.

## 2. The web interface (`app.py`)

* **Run experiment**: pick rounds, number of Sybils, attack, defense (and advanced attack/defense/DP settings) in the sidebar. Tick
  "also run the no-defense baseline" to get an automatic side-by-side comparison. Accuracy and attack success update live every round,
  with a summary (accuracy, label-flip or backdoor rate, Sybils caught / honest wrongly dropped).
* **Compare runs**: overlay any saved runs (UI runs and benchmark runs) and see a table of their metrics.
* **Saved benchmark**: the figures and tables from the two benchmarks.
* **How it works**: a short explanation for your audience.

Training runs on the machine that runs Streamlit; 3 rounds take ~15 s per run, 15 rounds about a minute on a laptop CPU.

## 3. Project layout

| File | What it does |
|---|---|
| `app.py` | Streamlit interface |
| `main_fed_avg_sybil.py` | training loop (DP noise → Sybil attack → defense → aggregation), logs accuracy, attack success, TPR/FPR |
| `defenses/sybil_filter.py` | **`filter_by_similarity`**, our similarity filter |
| `defenses/robust_agg.py` | Multi-Krum, median, trimmed mean, FoolsGold-style weighting |
| `attacks/sybil_attack.py` | attack strategies |
| `utils/options.py` | all CLI flags |
| `utils/data.py`, `utils/sampling.py`, `utils/dp.py`, `utils/backdoor.py` | data loading, IID/non-IID split, Laplace noise, backdoor trigger |
| `models/` | CNN / MLP, FedAvg, local training, evaluation |
| `benchmark.py`, `plot_results.py` | scale benchmark and figures |
| `benchmark_compare.py`, `plot_compare.py` | defense × attack comparison and figures |
| `tests/` | unit tests |

## 4. Our defense

For the N updates of a round (update = local weights − current global weights):

1. flatten every update to a vector `v_i`
2. `cos_ij = v_i·v_j / (‖v_i‖‖v_j‖)`
3. score `S_i`: `--filter_mode max` (default) → `S_i = max_{j≠i} cos_ij`; `mean` → `S_i = mean_{j≠i} cos_ij`
4. `S_i > τ` (`--filter_threshold`, default 0.9) → suspected Sybil
5. drop flagged updates, then aggregate the rest: **`--defense filter`** → FedAvg, **`--defense filter_median`** → coordinate-wise median (hybrid)

Two deliberate deviations from the plan text (both covered by unit tests):

* **The filter compares *updates* (w_i − w_global), not raw weights.** All clients start the round from the same global model, so raw
  weight vectors are all ≈ parallel (cosine ≈ 1) and nothing could be separated.
* **Default score is the *max* similarity, not the average.** With a few Sybils among N clients, a Sybil's *average* similarity is about
  (1 + (N−2)·ρ)/(N−1) with ρ ≈ 0, far below 0.9, so the average rule with τ = 0.9 never fires. The max rule is ≈1 for any pair of
  colluding Sybils. The literal average rule is still available (`--filter_mode mean`).

## 5. Attacks (`--attack`)

One adversary controls `--sybil` of the `--num_users` clients (fixed ids). It computes **one** malicious update per round and every Sybil
submits a perturbed copy (`--sybil_jitter`, default 0.2 → cosine between Sybils ≈ 0.96).

| `--attack` | behaviour | success metric |
|---|---|---|
| `labelflip` (default) | train on a shared shard with labels y → 9−y, update boosted ×`--sybil_boost` (3) | label-flip rate |
| `backdoor` | a fraction of the shard gets a 4×4 white trigger and label `--backdoor_target` (0), rest stays clean; boosted ("model replacement", Bagdasaryan et al. 2020) | backdoor success rate |
| `signflip` | negated, boosted honest update | accuracy |
| `random` | one shared random update | accuracy |
| `dpnoise` | the original repo's attack: tiny ε ⇒ huge independent noise (not similar, so similarity filters do not target it) | accuracy |
| *evasive* (setting, not a mode) | `--sybil_jitter 1.0` (and/or `--sybil_dp`): Sybils add independent noise so they are only ≈50% similar | as the base attack |

## 6. Comparison defenses (`--defense`)

| `--defense` | method | source |
|---|---|---|
| `none` | plain FedAvg | |
| `filter` | our similarity filter + FedAvg | this project |
| `filter_median` | our similarity filter, then coordinate-wise median of the survivors | this project |
| `krum` | Multi-Krum, assumed attackers f = 30% of the sampled clients | Blanchard et al., NeurIPS 2017 |
| `median` | coordinate-wise median | Yin et al., ICML 2018 |
| `trimmed` | coordinate-wise trimmed mean, 20% trimmed per side | Yin et al., ICML 2018 |
| `foolsgold` | similarity-based soft re-weighting. **Stateless per-round variant**: the original keeps per-client update history, which a random-sampling simulation rarely has | Fung et al. 2018 |

## 7. Results

### 7.1 Attack scale (label-flip, MNIST IID, 50 rounds, one seed)

| Sybils | Defense | Accuracy | Sybils caught (TPR) | Honest dropped (FPR) |
|---|---|---|---|---|
| 0% | none | 97.8% | – | – |
| 20% | none | 64.4% (very unstable) | – | – |
| 20% | our filter | 97.6% | 78% | 0% |
| 40% | none | 33.6% | – | – |
| 40% | our filter | 97.7% | 99% | 0% |

### 7.2 Defense × attack (30% Sybils, MNIST IID, 30 rounds, one seed, mean of last 10 rounds; clean run = 96.8%)

Each cell is accuracy % / attack-success % (label-flip rate, or backdoor rate for the backdoor column; lower is better for the second number).

| Defense | label-flip (colluding) | backdoor (colluding) | label-flip, evasive Sybils |
|---|---|---|---|
| none (FedAvg) | 53.0 / 42.0 | 95.8 / 99.9 | 55.1 / 39.7 |
| similarity filter (ours) | 96.7 / 0.6 | 96.8 / 0.5 | 55.1 / 39.7 |
| **filter + median (ours)** | 96.6 / 0.6 | 96.7 / 0.4 | **77.7 / 18.2** |
| Multi-Krum | 78.8 / 15.8 | 89.3 / 67.8 | 78.6 / 17.2 |
| median | 76.3 / 17.8 | 93.9 / 73.1 | 77.7 / 18.2 |
| trimmed mean | 57.5 / 37.6 | 95.7 / 99.9 | 58.5 / 37.0 |
| FoolsGold-style | 96.7 / 0.6 | 96.7 / 0.5 | 54.9 / 40.0 |

What to take from it:

* **Against colluding Sybils** the similarity-based methods (our filter, the hybrid, FoolsGold-style) restore clean accuracy and remove the
  attack. Krum and median only partly help; the 20% trimmed mean cannot remove a 30% attacker group.
* **Accuracy alone hides a backdoor**: without a defense the backdoor model is still 96% accurate on clean images, but 99.9% of
  triggered images are classified as the attacker's label. Always report the backdoor rate.
* **Against evasive Sybils the pure similarity filter fails** (it flags nobody, so it equals no defense), and so does FoolsGold-style.
  Krum and median are less affected because they do not rely on similarity. The **hybrid filter + median** keeps the full benefit
  against colluding Sybils and falls back to median-level robustness against evasive ones.
* Even the best defense in the evasive column is far from the clean 96.8%. Detecting stealthy Sybils is not solved by these methods.

## 8. Known limitations (state these in the report)

* A **single** Sybil in a round has no partner, so a pairwise similarity filter cannot see it (TPR is below 100% at low Sybil share).
* The evasive attack is a simple one (independent Gaussian noise); stronger adaptive attacks against each defense were not tried.
* Single seed, MNIST, one model, 30–50 rounds. For error bars run `python benchmark.py --seeds 1 2 3` (the comparison benchmark takes `--seed`).
* With non-IID data honest clients also look alike; re-tune τ (`--filter_threshold`).
* The FoolsGold-style variant is simplified (no history), so it is not a faithful reproduction of the paper's results.

## 9. Mapping to the team plan

| Member | Part |
|---|---|
| 1 Pipeline & setup | `utils/` (data, sampling, options), `app.py`, `run_all.sh`, `requirements.txt` |
| 2 Attack | `attacks/sybil_attack.py`, `utils/backdoor.py` |
| 3 Defense | `defenses/`, defense dispatch in `main_fed_avg_sybil.py`, `tests/` |
| 4 Benchmark & visuals | `benchmark.py`, `benchmark_compare.py`, `plot_*.py`, `results/` |

## 10. Notes for the report

* Base repo: ITSEG-MQ/Sybil-Attacks-and-Defense-on-DP-FL (no license file; cite it and keep the attribution if you publish).
* Papers behind the added methods: FoolsGold (Fung et al. 2018), Krum (Blanchard et al., NeurIPS 2017), median / trimmed mean
  (Yin et al., ICML 2018), backdoor / model replacement (Bagdasaryan et al., AISTATS 2020).
* Changed from the base repo: in-memory data loading (faster), MLP returns log-probs, new attack/defense modules, logging, web app.
