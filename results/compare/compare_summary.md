# Defense x attack comparison (MNIST, IID, 100 clients, 30% Sybils, 30 rounds, last-10-round mean)

Clean run (no attack, no defense): **96.82%** accuracy.

Each cell: accuracy % / attack-success % (label-flip rate, or backdoor rate for the backdoor attack).

| Defense | label-flip (colluding) | backdoor (colluding) | label-flip, evasive (noisy Sybils) |
|---|---|---|---|
| no defense (FedAvg) | 53.0 / 42.0 | 95.8 / 99.9 | 55.1 / 39.7 |
| similarity filter (ours) | 96.7 / 0.6 | 96.8 / 0.5 | 55.1 / 39.7 |
| filter + median (ours) | 96.6 / 0.6 | 96.7 / 0.4 | 77.7 / 18.2 |
| Multi-Krum | 78.8 / 15.8 | 89.3 / 67.8 | 78.6 / 17.2 |
| median | 76.3 / 17.8 | 93.9 / 73.1 | 77.7 / 18.2 |
| trimmed mean | 57.5 / 37.6 | 95.7 / 99.9 | 58.5 / 37.0 |
| FoolsGold- style | 96.7 / 0.6 | 96.7 / 0.5 | 54.9 / 40.0 |
