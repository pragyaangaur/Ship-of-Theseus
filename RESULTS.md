# Results of the 7B run

Kaggle kernel `pragyaangaur/theseus-main` version 1, run on 6 November 2026 on two T4s. It finished all 261 evaluations (4 controls, 240 plank hybrids, 8 part-kind hybrids and 9 blends) in 4.9 hours with no non-finite logit. The raw output is in `results/main/`. Numbers below come from `summary.json` and `table.json`, which the run wrote itself. Here ι is the self-report score and φ the behavioural fingerprint, both scaled so that Qwen (ship A) is 0 and DeepSeek-R1-Distill (ship B) is 1.

## Gates

All four gates passed. Under amendment 1 the own forward pass agreed with Hugging Face on every decisive position (32 of 32 for A, 26 of 26 for B), and the largest logit differences were 0.79 and 0.42. Under the original wording gate 1 would have failed for B, which matched on 50 of 51 positions (98.0%), so the result rests on amendment 1, which was written after the pilot and before this run. The probes separated the parents by 33.1 nats (A template) and 26.6 nats (B template) and the fingerprint by 2.35 nats per token.

The parents scored WikiText perplexity 12.4 (A) and 43.6 (B) and ARC-Easy accuracy 0.95 and 0.83. Giving A the rotary base of B raised its perplexity to 13.6, and giving B the base of A raised it to 63.8.

## Preregistered hypotheses

| Hypothesis | Prediction | Result | Verdict |
| --- | --- | --- | --- |
| H1, tipping point | Logistic beats linear by ΔBIC of at least 10 for forward and backward orders under both templates | ΔBIC −1.2 (A template, forward), 11.6 (A, backward), 56.1 (B, forward), 1.4 (B, backward) | Not supported. Two of the four curves have a sharp switch and two are gradual |
| H2, identity near the output | k50 is lower in the backward order than in the forward order | A template: k50 22.2 forward against 5.1 backward, difference interval [14.6, 18.6]. B template: 26.5 against 7.4, interval [17.1, 21.2] | Supported under both templates |
| H3, Hobbes's twins | No direction predicted. Report D = ι(M) + ι(not M) − 1 | Mean D 0.152, 95% interval [0.125, 0.177] over 119 pairs (238 distinct masks, because two masks occur in two orders each). The additive plank model explains R² 0.85 of ι | Identity is mostly additive, with a reliable lean toward B. Two twins together claim to be DeepSeek more than one whole ship would |
| H4, saying and doing come apart | At least 10% of seaworthy plank hybrids have abs(ι − φ) > 0.5 | 0 of 64 seaworthy hybrids | Not supported |
| H5, the middle sinks | Highest intermediate perplexity is more than twice the worse parent's in forward and backward orders, and blends beat plank swaps at matched fractions on most of 9 points | Highest perplexity 29,904 (forward) and 86,002 (backward) against a limit of 87. Blends lower on 9 of 9 points, never above 48.2 | Supported |

In the plank influence map the last five decoder blocks, the final norm and the output head carry the largest positive weights on ι (about 0.06 to 0.10 each), and the second block (layer 0) carries 0.11. The same additive model explains only R² 0.17 of log perplexity, so whether a hybrid works depends on which planks sit next to each other, unlike what it claims to be.

## Exploratory observations

These were not preregistered.

- **The middle ship is nobody.** Under A's template, 1,745 of the 1,920 free self-reports from plank hybrids named neither Qwen nor DeepSeek, against 36 naming Qwen and 139 naming DeepSeek. 2,297 of all 3,840 replies collapsed into repetition. The non-degenerate ones mostly say "I'm an AI assistant" with no name. One forward-order hybrid with 26 of 31 planks from B said it was "created by {{ company }}". A blend at a = 0.5 called itself "Assistant AI".
- **The forced-choice probe and free speech disagree on blends.** A 10% blend scored ι = 0.09, close to Qwen, yet it named Qwen in only 1 of its 16 free replies, where Qwen itself does in 11 of 16. The forced choice measures which name is more likely when the model is made to give one. It does not measure whether the model volunteers a name.
- **Thinking leaks across.** 275 of the 1,920 plank-hybrid replies under A's template emitted DeepSeek's `<think>` or `</think>` tokens, although nothing in that template asks for them.
- **The pilot misled on blends.** In the 1.5B pilot blends broke badly (perplexity above 2,000 at a = 0.4), while at 7B they stayed within the range of the parents.
- **Random orders sink hardest.** One random order kept perplexity above 10⁵ for its first 12 steps, peaking at 3.4 × 10⁸. Contiguous orders recover faster.
