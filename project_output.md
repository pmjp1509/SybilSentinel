Your current settings (DP-noise attack, Multi-Krum, boost 4) aren't the best ones to show. The DP-noise attack is the original repo's, and I haven't benchmarked it. The boost and stealth sliders don't affect it at all. Here are three runs that tell the story, in this order.

Leave everything under "Advanced" at its default: boost 3, stealth 0.2, tau 0.90, DP noise on, epsilon 8, local epochs 2, IID on, seed 1. Keep "Also run the no-defense baseline" ticked.

| Run | Rounds | Sybils | Attack | Defense | What to look at |
|---|---|---|---|---|---|
| **1. Attack and defense** | 20 | 30 | Label-flip | Similarity filter | Without a defense accuracy collapses. With the filter it stays near the clean level, and the summary shows Sybils caught and 0% honest clients dropped. |
| **2. Hidden backdoor** | 20 | 30 | Backdoor | Similarity filter | Look at the backdoor success rate, not accuracy. Undefended, the model is still accurate on normal images but nearly every triggered image gets the attacker's label. |
| **3. The weakness** | 20 | 30 | Label-flip, with stealth set to **1.0** | Similarity filter, then again with **Filter + median hybrid** | The plain filter fails, because the Sybils no longer look alike. The hybrid recovers part of the damage. |

For run 3, first run the plain filter with stealth 1.0, then change only the defense to the hybrid and run again.

- **Time:** on my 2-core test machine it took about 5 seconds per round per run, so each pair of runs takes roughly 3–4 minutes at 20 rounds. Your machine may differ.
- **What to expect:** in my 30-round benchmark at 30% Sybils, the undefended label-flip run reached about 53% accuracy against about 97% with the filter. With stealth 1.0, the plain filter stayed at about 55%, and the hybrid reached about 78%. Your 20-round numbers will differ somewhat. They are single-seed runs, so rounds will bounce around.
- **Comparing several defenses:** the checkbox only compares your chosen defense against no defense. To add Krum, median and the others, run each one separately and overlay them in the **Compare runs** tab.
- **Saved-run naming:** running the same attack, Sybil count, defense and seed again overwrites the earlier file, even if the stealth value differs. For runs 3a and 3b the defenses differ, so both are kept. If you repeat a setting with a different stealth value, change the seed first to keep both runs.