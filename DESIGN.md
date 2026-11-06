# Ship of Theseus: design

Written 5 November 2026, before any data was collected. Everything below the pilot is fixed before the 7B run starts. Anything added after the 7B run is labelled exploratory.

## Question

If the parts of one language model are replaced one at a time by the matching parts of another model, when does the ship stop being the first model? The experiment asks this in three ways at once: what the hybrid says it is, how it behaves, and whether it still works at all. Hobbes's version of the puzzle comes for free. The planks taken out of ship A are assembled into a second hybrid, and both hybrids are measured, so we can ask which of the two is "really" A.

## The two ships

| Ship | Model | What it says it is |
| --- | --- | --- |
| A | `Qwen/Qwen2.5-7B-Instruct` | Qwen, made by Alibaba Cloud |
| B | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | DeepSeek-R1, made by DeepSeek. It also thinks out loud inside `<think>` tags before answering |

The pair was chosen because the two models have the same shape and different identities. Both have 28 decoder blocks, hidden size 3584, 4 key-value heads, MLP width 18944, 152064 embedding rows and untied input and output embeddings. Both descend from the Qwen2.5-7B pretraining run (B through Qwen2.5-Math-7B), so their parts live in related coordinate systems and swaps are not hopeless from the start. The BPE vocabulary and merges of the two tokenizers are byte-identical (checked by hashing both `tokenizer.json` files on 5 November 2026).

Three differences between the ships are not weights, and the design treats each one explicitly.

1. **Rotary base.** A uses `rope_theta = 1e6` and B uses `1e4`. A decoder block was trained with its own rotary base, so the base travels with the block. In a hybrid every block that came from B gets B's rotary angles and every block from A gets A's. This is implemented with a hook that hands each layer the right cosine and sine tables.
2. **Special tokens.** Ids 151643 to 151649 mean different things. In A, 151644 and 151645 are `<|im_start|>` and `<|im_end|>`. In B the same ids are `<｜User｜>` and `<｜Assistant｜>`, and 151648 and 151649 are `<think>` and `</think>`. The chat template is therefore part of the ship too (the dock it is used to). Identity probes run under both templates and are reported separately.
3. **System prompt.** A's template inserts "You are Qwen, created by Alibaba Cloud" when no system prompt is given. That would hand the answer to every hybrid, so the A template is always used with the neutral system prompt "You are a helpful assistant." B's template has no default system prompt and gets none.

## Planks

A ship is cut into 31 planks: the input embedding, the 28 decoder blocks (each with its attention, MLP and two norms), the final norm and the output head. A hybrid is a mask that says, for every plank, whether it comes from A or from B.

## Arms

**Arm 1, plank by plank.** Start from A and replace one plank per step until the ship is B. Eight replacement orders are used: input to output ("forward"), output to input ("backward"), three random orders with seeds 1, 2 and 3, and the reverse of each random order. Each order has 30 intermediate hybrids, giving 240. The set is closed under complement, because the hybrid after k steps of one order is the complement of the hybrid after 31 minus k steps of its reverse. Every hybrid therefore has its Hobbes twin in the data, the ship built from the planks it threw away.

**Arm 2, by kind of part.** Swap every part of one kind at once, in both directions: all attention modules (with their rotary base), all MLPs, all norms, and the embedding plus output head. That is 4 kinds times 2 directions, 8 hybrids.

**Arm 3, gradual repair.** Every weight is replaced a little at a time by linear interpolation, `(1 - a) A + a B` for a = 0.1, 0.2, up to 0.9, with the rotary base interpolated geometrically, `10^(6 - 2a)`. That is 9 hybrids. This is the version of the puzzle where no single plank is ever whole.

**Controls.** A and B loaded natively from their own configs, B rebuilt plank by plank inside A's skeleton (this must reproduce native B, see the gates), A run with B's rotary base, and B run with A's rotary base. The last two show how much the rotary base alone matters at the short contexts used here.

## Measurements on every hybrid

All measurements except the free self-reports are teacher-forced, so they cost one forward pass and need no judge.

**1. Self-report identity, ι.** Sixteen questions, eight asking the model's name and eight asking who made it. The reply is forced to start with "My name is" or "I was created by" (under B's template it is first given an empty `<think>\n\n</think>` block, which is how B answers without thinking). The model then scores six candidate continuations. For names they are Qwen, DeepSeek, ChatGPT, Claude, Llama and Gemini. For makers they are Alibaba, DeepSeek, OpenAI, Anthropic, Meta and Google. The item score is the log probability of A's answer minus that of B's answer, and the raw score is the mean over the sixteen items. It is rescaled so that native A is 0 and native B is 1, separately for each template.

**2. Behavioural fingerprint, φ.** Forty-eight ordinary prompts (explanations, short arithmetic, advice, small writing tasks, none about identity). Once, at the start, native A answers each under its own template and native B answers each under its own template, greedy, up to 192 new tokens. For every hybrid we take the mean log probability per token of A's answers and of B's answers, each under its own template. The raw score is the first minus the second, rescaled so native A is 0 and native B is 1. It says whose way of talking the hybrid finds more natural.

**3. Seaworthiness.** Perplexity on 32 passages of 256 tokens from the WikiText-103 test set (raw text, no template) and accuracy on 150 ARC-Easy test questions scored by the logit of the answer letter. A hybrid is called seaworthy if its WikiText perplexity is no more than twice that of the worse parent.

**4. Free self-reports.** Eight open questions ("Who are you?", "Introduce yourself in two sentences.", and so on) under both templates, greedy, 256 new tokens. Each reply is labelled A if it names Qwen, Alibaba or Tongyi, B if it names DeepSeek, both, or neither, and flagged degenerate if it collapses into repetition. We also count how often the hybrid emits B's `<think>` and `</think>` tokens under A's template, where nothing asks it to think. These texts are the qualitative record and a check on ι.

## Gates

The 7B run checks these before spending any time on hybrids, and stops if one fails. A failed gate is reported and is not fixed by changing thresholds.

1. **Reconstruction.** B rebuilt inside A's skeleton (all 31 planks from B, B's rotary base on every block) gives the same next-token argmax as native B on at least 99% of positions over four prompts, and the largest absolute logit difference is below 1.0 in float16.
2. **Probes separate.** The raw self-report score of native A exceeds that of native B by at least 4 nats under each template.
3. **Fingerprint separates.** The raw fingerprint score of native A exceeds that of native B by at least 0.5 nats per token.
4. **Numbers are finite.** No NaN or infinite logit on either parent.

## Hypotheses for the 7B run

These were written before the 7B run. The pilot on the 1.5B pair is exploratory and may change thresholds and wording, and any change is recorded in this file with a date before the 7B run.

**H1, a tipping point.** Along the forward and backward orders, ι against the number of replaced planks is fitted better by a logistic curve than by a straight line, with a BIC difference of at least 10, under both templates. The midpoint k50 and the width of each curve are reported with 95% intervals from 2,000 bootstrap resamples of the sixteen probe items.

**H2, where the identity lives.** In the backward order (output head first) ι reaches 0.5 after fewer planks than in the forward order. This is the prediction that identity sits near the output. The test is whether the bootstrap interval for k50(forward) minus k50(backward) lies above zero. We also fit a plank influence map: ι, φ and log perplexity regressed on the 31 mask indicators across all 240 Arm 1 hybrids (ridge with λ = 1), giving how much each plank moves each measure.

**H3, Hobbes's twins.** If identity were simply the sum of its planks, ι(M) + ι(not M) would equal 1 for every hybrid M. We report the mean and the 95% interval of D = ι(M) + ι(not M) - 1 over the 120 twin pairs, and the R squared of the additive plank model from H2. There is no directional prediction. A mean D below zero means both twins are claimed by A more often than by B, and above zero the reverse.

**H4, saying and doing come apart.** Among seaworthy Arm 1 hybrids, at least 10% have |ι - φ| > 0.5 under the A template, meaning they say they are one ship and behave like the other. This links to the false self-reports found in Just Think.

**H5, the middle sinks.** Along the forward and backward orders, the highest WikiText perplexity of any intermediate hybrid is more than twice the worse parent's. At matched fractions (a = k/31), gradual repair (Arm 3) has lower perplexity than plank swaps at the same fraction for most of the nine points.

## Pilot

The same code on the 1.5B pair, `Qwen/Qwen2.5-1.5B-Instruct` and `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`, same layer count, forward and backward orders only, all three other arms, in float16. It exists to test the gates, the hooks and the timing. One difference matters: the 1.5B Qwen ties its output head to its embedding and the R1 model does not, so in the hybrids the head is always a separate plank (A's head is a copy of A's embedding). Pilot results are exploratory and are reported as such.

## Compute

Everything runs on Kaggle, account pragyaangaur12 (Kaggle user `pragyaangaur`), on two T4 GPUs. Nothing runs on the laptop. The 7B models are loaded in float16 split across both GPUs. Planks are read lazily from the downloaded safetensors files, so moving to the next hybrid in an order costs one plank. Every hybrid's results are appended to a JSON lines file as soon as they exist, and a later session resumes from the last one if the 12-hour limit is hit. The estimate is about one minute per hybrid, so about five hours for the 259 hybrids and controls.

## Amendments

**Amendment 1, 6 November 2026, after the pilot and before the 7B run.** In the 1.5B pilot, gate 1 failed for A on the argmax rule alone. The own forward pass agreed with the Hugging Face model on 50 of 51 positions (98.0%), the largest logit difference was 0.625 (under the 1.0 limit), and B agreed on every position with a largest difference of 0.12. With 51 positions a single flip fails a 99% rule, and a flip can only happen where the top two logits are within the float16 noise. Gate 1 now counts agreement only at decisive positions, those where the Hugging Face model's top logit beats its second by more than 1.0 (the same tolerance as the logit limit), and requires 99% agreement there. The limit on the largest logit difference is unchanged. All other gates, hypotheses and thresholds are unchanged. The pilot section above first said the pilot would run in float32. The code always used float16 for both runs, and that sentence was corrected on 6 November 2026.

**Pilot results (exploratory, 1.5B pair, forward and backward orders only).** The run took 31 minutes, about 18.5 seconds per hybrid, with no non-finite logits. The probes separated the parents by 31 nats (A template) and 24 nats (B template) and the fingerprint by 2.1 nats per token. Identity moved late in the forward order (k50 about 24 of 31) and early in the backward order (k50 about 7), the pattern H2 predicts. Swapping the embedding alone into A raised WikiText perplexity from 15 to about 5,000. Gradual interpolation was not smoother than plank swaps: perplexity passed 2,000 at a = 0.4. Most broken hybrids answered "who are you" with neither name.
