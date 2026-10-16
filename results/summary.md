# Results summary (MNIST, IID, 100 clients)

| Sybil % | Defense | Final acc (last-5 mean) | Drop vs. clean baseline | Label-flip rate | TPR | FPR |
|---|---|---|---|---|---|---|
| 0% | none | 97.78% | +0.00 pts | 0.3% | - | - |
| 0% | filter | 97.78% | +0.00 pts | 0.3% | - | 0.0% |
| 20% | none | 64.40% | -33.38 pts | 69.5% | - | - |
| 20% | filter | 97.58% | -0.20 pts | 0.4% | 78% | 0.0% |
| 40% | none | 33.55% | -64.23 pts | 84.0% | - | - |
| 40% | filter | 97.65% | -0.13 pts | 0.4% | 99% | 0.0% |

TPR = fraction of Sybil submissions the filter dropped; FPR = fraction of honest submissions dropped.
Label-flip rate = % of test images classified as the attacker's target label (9 - y).
