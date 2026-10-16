# Project Explained: Sybil Attacks on Federated Learning

A plain-language guide to what this project is, what the base code does, what we added, and what the results mean.

---

## 1. The big picture in 30 seconds

- **Federated Learning (FL):** many clients (phones, hospitals) train one shared model together **without sharing their data**. Each client trains locally and sends only its model update to a server. The server averages the updates (**FedAvg**) into a new global model. This repeats for many rounds.
- **Sybil attack:** one attacker creates **many fake clients** ("Sybils"). Because the server averages all clients equally, many fake clients means a lot of voting power, so the attacker can wreck or steer the model.
- **Our project:** (1) simulate this attack, (2) add a **defense** that spots fake clients because they behave too alike, (3) **measure** how well the defense works.

---

## 2. Background terms

| Term | Meaning here |
|---|---|
| **Client** | One simulated user holding a small slice of MNIST (handwritten digits). 100 clients in total. |
| **Round** | Server picks 10 random clients, they train, the server averages their results. We run 50 rounds. |
| **FedAvg** | New global model = plain average of the clients' models. |
| **Update** | `client model − global model`: the direction the client wants to move the model. |
| **DP noise (Laplace)** | Each honest client adds random noise to its model before sending, to protect privacy. (Taken from the base repo.) |
| **Sybil** | A fake client controlled by the attacker. |
| **Cosine similarity** | Number from −1 to 1 saying whether two updates point in the same direction. 1 = identical direction, 0 = unrelated. |

---

## 3. What the base repo does (ITSEG-MQ / Sybil-Attacks-and-Defense-on-DP-FL)

Based on its code (I read `main_fed_avg_sybil.py` and its helper files):

1. Loads MNIST (or CIFAR-10) and splits it across 100 clients.
2. Runs the FedAvg loop: pick clients, train locally, average.
3. **Differential privacy:** every honest client adds Laplace noise to its weights (privacy level epsilon = 8, i.e. a small amount of noise).
4. **Its Sybil attack:** the "Sybil" clients use a tiny epsilon (0.1), which means they add **huge noise** to their updates. These junk updates get averaged in and drag the model down.
5. It saves loss/accuracy logs and a loss plot.

Its weaknesses, which is where our work comes in:

- The attack is just "send noisy garbage". The fake clients are **not coordinated**, so they look nothing alike.
- The main script has **no similarity-based defense**: nothing catches colluding clients.
- No benchmark across attack sizes, no detection metrics (how many fake clients were caught or wrongly blocked), no comparison plots.

The repo also contains other scripts (Krum, trimmed mean, a sybil-defense script). We did not use or change those.

---

## 4. What we added

### 4.1 A realistic, coordinated Sybil attack (`attacks/sybil_attack.py`)
One attacker controls the Sybils, so they act together:

- The attacker computes **one** malicious update per round (default: train on **flipped labels**, so a "3" is taught as a "6", then multiply the update by 3 to make it stronger).
- Every Sybil submits a **slightly jittered copy** of it. They are ~96% similar to each other.
- Other modes: `signflip`, `random`, and the original `dpnoise` attack from the base repo.

### 4.2 The defense: cosine-similarity Sybil filter (`defenses/sybil_filter.py`), our main contribution
**Intuition:** honest clients have different data and different privacy noise, so their updates point in different directions. Sybils from one attacker point almost the same way. So **update too similar to someone else's → suspicious → drop it.**

Each round, before averaging:

1. Take each client's update (`client weights − global weights`) and flatten it into one long vector.
2. Compute the cosine similarity between every pair of clients.
3. Give each client a score = its **highest** similarity to any other client.
4. If the score is above **0.9**, flag it as a Sybil and drop it.
5. Run FedAvg on the remaining clients.

This sits **on top of** the base repo's DP noise, so the combined defense is **noise (privacy) + similarity filter (anti-Sybil)**.

Two choices where we deliberately differ from the first plan:

- **We compare updates, not raw weights.** All clients start from the same global model, so raw weights are all nearly identical and the test would flag everyone.
- **We use the maximum similarity, not the average.** With only 2 Sybils among 10 clients, a Sybil's *average* similarity is low because it is similar to only 1 of 9 others. The *maximum* correctly shows ≈1. (The average version is still available with `--filter_mode mean`.)

### 4.3 Benchmark and plots (`benchmark.py`, `plot_results.py`)
Runs 0%, 20% and 40% Sybils, each with and without the filter, then produces:

- accuracy per round,
- final accuracy vs Sybil share,
- detection quality (TPR and FPR),
- a summary table.

### 4.4 Engineering extras
Faster in-memory data loading, a CLI flag `--enable_filter`, unit tests for the filter, a README and `run_all.sh`.

---

## 5. One training round, step by step

```
Server picks 10 clients
        │
        ├── honest client: train on own data → add DP noise → send
        └── Sybil client : copy attacker's malicious update (+ small jitter) → send
        │
        ▼
FILTER (our addition): cosine similarity between all pairs of updates
        │   similarity score > 0.9 ?  → drop that client
        ▼
FedAvg on the remaining updates → new global model → test accuracy
```

---

## 6. Results (MNIST, IID, 100 clients, 50 rounds, one seed)

| Sybils | Defense | Accuracy | Sybils caught (TPR) | Honest wrongly dropped (FPR) |
|---|---|---|---|---|
| 0% | none | 97.8% | – | – |
| 20% | none | 64.4% (swings wildly) | – | – |
| 20% | **filter** | **97.6%** | 78% | 0% |
| 40% | none | 33.6% | – | – |
| 40% | **filter** | **97.7%** | 99% | 0% |

How to read it:

- **Without defense**, the attack collapses accuracy. At 40% Sybils the model is mostly broken.
- **With the filter**, accuracy stays within about 0.2 points of the clean baseline, and **no honest client was wrongly dropped**.
- **TPR** = share of Sybil submissions caught. **FPR** = share of honest submissions wrongly dropped.
- **Why TPR is 78%, not 100%, at 20%:** if only **one** Sybil is picked in a round, it has no partner to look similar to, so the filter cannot see it. In our runs those single-Sybil rounds did little damage; the harmful rounds are the ones with 2 or more Sybils, which the filter catches.

The charts are in `results/`: `acc_vs_rounds.png`, `acc_vs_sybil.png`, `detection.png`.

---

## 7. Limitations (say these yourself in the viva or report)

1. **Lone Sybil:** invisible to a pairwise similarity filter.
2. **Smart attacker:** Sybils that add lots of independent noise (`--sybil_jitter 1.0` or `--sybil_dp`) become less similar to each other and may slip past a fixed 0.9 threshold. We did not run this experiment; it would be a good extension.
3. **Non-IID data:** honest clients also look more alike, so the threshold may need re-tuning.
4. **One seed, MNIST only:** run `python benchmark.py --seeds 1 2 3` for more reliable curves.
5. The attack's strength (label flip with ×3 boost, 96% internal similarity) is our own design choice.

---

## 8. Files and what each does

| File | Role |
|---|---|
| `main_fed_avg_sybil.py` | Main training loop: DP noise → attack → filter → FedAvg → log |
| `defenses/sybil_filter.py` | **Our defense** (`filter_by_similarity`) |
| `attacks/sybil_attack.py` | Sybil attack strategies |
| `benchmark.py` | Runs all scenarios, then plots |
| `plot_results.py` | Makes charts and `summary.md` |
| `utils/options.py` | All command-line flags |
| `utils/data.py`, `utils/sampling.py` | Load MNIST, split IID/non-IID |
| `utils/dp.py` | Laplace privacy noise (from base repo) |
| `models/` | CNN/MLP models, FedAvg, local training, testing |
| `tests/test_filter.py` | Unit tests |

---

## 9. Key commands

```bash
pip install -r requirements.txt
python tests/test_filter.py
python main_fed_avg_sybil.py --sybil 20                    # attack only
python main_fed_avg_sybil.py --sybil 20 --enable_filter    # attack + our defense
python benchmark.py                                        # everything + plots
```

---

## 10. Three-sentence summary

> Federated learning averages updates from many clients, so an attacker who creates many fake clients can corrupt the model. The base repo simulates a simple noisy-update Sybil attack inside a differentially-private FL setup; we added a **coordinated** attack and a **cosine-similarity filter** that drops clients whose updates are suspiciously alike, working together with the DP noise. In our MNIST experiments the filter restored accuracy from 34–64% back to about 97.6% with zero honest clients wrongly dropped, but it cannot detect a single isolated Sybil and could be evaded by an attacker who adds enough independent noise.

---

**Credits:** base code from ITSEG-MQ/Sybil-Attacks-and-Defense-on-DP-FL; the similarity-filter idea is inspired by FoolsGold (Fung et al.). Cite both in your report.

---

## 11. Version 2 additions: web app, more attacks, comparison against known defenses

### What was added
- **Web interface** (`streamlit run app.py`): choose the attack, the defense and the settings, press Run, and watch accuracy and attack success update live. It can automatically run a no-defense baseline next to your chosen defense, compare any saved runs, and show the saved benchmark figures.
- **Backdoor attack**: the Sybils teach the model that a small white square means "digit 0" while keeping normal accuracy high. Its success is measured by the *backdoor success rate*.
- **Evasive Sybils**: Sybils add independent noise so they no longer look alike. This tests the weak spot of the similarity filter.
- **Comparison defenses** from the literature: Multi-Krum (Blanchard 2017), coordinate-wise median and trimmed mean (Yin 2018), and a FoolsGold-style re-weighting (Fung 2018, simplified: no per-client history).
- **A hybrid defense of our own**: similarity filter first, then the median of whatever survives.

### Results (30% Sybils, 30 rounds, one seed; accuracy % / attack success %; clean run = 96.8% accuracy)

| Defense | label-flip | backdoor | evasive label-flip |
|---|---|---|---|
| none | 53.0 / 42.0 | 95.8 / 99.9 | 55.1 / 39.7 |
| similarity filter (ours) | 96.7 / 0.6 | 96.8 / 0.5 | 55.1 / 39.7 |
| filter + median (ours) | 96.6 / 0.6 | 96.7 / 0.4 | 77.7 / 18.2 |
| Multi-Krum | 78.8 / 15.8 | 89.3 / 67.8 | 78.6 / 17.2 |
| median | 76.3 / 17.8 | 93.9 / 73.1 | 77.7 / 18.2 |
| trimmed mean | 57.5 / 37.6 | 95.7 / 99.9 | 58.5 / 37.0 |
| FoolsGold-style | 96.7 / 0.6 | 96.7 / 0.5 | 54.9 / 40.0 |

### How to read it
1. **Colluding Sybils:** the similarity-based methods (ours, the hybrid, FoolsGold-style) bring accuracy back to the clean level and almost remove the attack. Krum and median help only partly. The 20% trimmed mean cannot remove a 30% attacker group.
2. **Accuracy hides backdoors:** with no defense the backdoored model is still 96% accurate on normal images, yet 99.9% of images carrying the trigger are classified as the attacker's label. That is why the backdoor rate is reported separately.
3. **Evasive Sybils break the pure similarity filter:** it flags nobody, so it behaves exactly like no defense (the same 55.1 / 39.7). FoolsGold-style fails the same way. Krum and median do not rely on similarity, so they hold up better.
4. **The hybrid covers both cases:** it keeps the similarity filter's full benefit on colluding Sybils and falls back to median-level robustness on evasive ones (77.7 / 18.2).
5. **Not solved:** even the best defense against evasive Sybils is far below the clean 96.8%.

### Extra limitations
One seed, MNIST only. The evasive attack is simple (Gaussian noise). The FoolsGold-style version is stateless, so it is not a faithful reproduction of the paper. Krum assumes an attacker share of 30%.
