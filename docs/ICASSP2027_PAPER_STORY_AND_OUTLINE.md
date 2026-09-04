# ICASSP 2027 paper story and outline

Status: working paper plan; frozen Noisy TEST results pending  
Updated: 2026-09-04

## Recommended title

> **Repair Before Grounding: Target-Consistent Acoustic Evidence for LLM-Based Target Speech Extraction**

This is the recommended title because it states the central design principle,
the technical object being repaired, the LLM connection, and the speech
front-end task without claiming that the LLM branch wins every metric.

Alternative titles:

1. **Speaker-Consistent Grounding for Confusion-Resilient LLM-Based Target Speech Extraction**
2. **When Grounding Selects the Wrong Speaker: Confusion Repair for LLM-Based Target Speech Extraction**
3. **Target-Consistent Candidate Selection and Code-Space Grounding for LLM-Based Speech Extraction**

## One-sentence claim

> LLM-based target speech extraction fails when the decoder is grounded to a
> fluent but wrong-speaker waveform; repairing the acoustic evidence with
> target-consistent candidate selection before code-space grounding suppresses
> speaker swaps and yields a substantially better reliability--naturalness
> trade-off.

The paper must not claim that the LLM output universally outperforms the direct
waveform. The evidence supports two useful operating points:

- **Pool-D direct** is the fidelity-oriented output with the best WER and
  content preservation.
- **Fixed CSG** is the reliable generative output: compared with matched
  ungrounded Q-Full decoding, it reduces decoder drift and speaker confusion
  while retaining most of the LLM resynthesis naturalness gain.

## The story reviewers should remember

### The problem

Speech LLM front ends often assume that the acoustic evidence supplied to the
decoder is trustworthy. In TSE this assumption can fail catastrophically: the
primary separator may output intelligible speech from the interfering speaker.
An LLM grounded to that waveform is then encouraged to preserve the wrong
identity. This is an evidence-interface failure, not merely a low-MOS artifact.

### The insight

**Grounding cannot rescue evidence that belongs to the wrong speaker.** The
system should first establish target identity, then constrain generation.

### The method

1. Construct five complementary frozen TSE candidates from the same mixture
   and target enrollment: four deterministic enrollment views from the primary
   extractor plus one TF-map/context-conditioned alternative expert.
2. Select the candidate with maximum frozen ECAPA similarity to the complete
   target enrollment. No clean target, interferer, transcript, WER, or oracle
   score is available to the selector.
3. Tokenize the repaired evidence and ground Q-Full decoding in the native
   `3^8` FSQ space using CSG. GNR is an ablation, not the headline method,
   unless the final frozen TEST contradicts the DEV hierarchy.

### The evidence chain

The results must establish the following claims in this order:

1. **The failure is real:** Primary TSE has measurable content and acoustic
   speaker switches.
2. **A correct candidate exists:** candidate diversity recovers most frozen
   primary-swap cases.
3. **The deployable selector finds it:** enrollment-only selection approaches
   the clean-reference oracle while almost never damaging primary-correct
   controls.
4. **Evidence quality matters to the LLM:** replacing primary evidence with
   Pool-D evidence improves the matched Q-Full branch.
5. **Grounding matters after repair:** fixed CSG improves over matched
   ungrounded decoding on content and speaker reliability.
6. **The trade-off is explicit:** direct evidence maximizes fidelity; grounded
   resynthesis supplies higher predicted naturalness at some remaining content
   cost.
7. **The conclusion generalizes:** the one-run frozen Noisy TEST, paired
   statistics, per-SNR analysis, and cost reporting support the same hierarchy.

Clean TEST already gives a strong anchor: Primary WER `13.68%` falls to
`6.54%` with Pool-D direct, while acoustic switch falls from `6.98%` to
`0.78%`. On frozen primary swaps, the deployable selector recovers `355/398`
(`89.20%`), only five trials below the oracle, and causes only `2/5,588`
(`0.0358%`) candidate-level wrong-speaker regressions on controls.

Noisy DEV currently supports the trade-off claim: Pool-D direct has the best
WER (`46.75%`), while fixed CSG improves matched ungrounded Q-Full from
`66.87%` to `62.66%` WER and from `2.99%` to `1.27%` acoustic switches. Its
UTMOS (`3.047`) remains far above Pool-D direct (`1.930`) but below ungrounded
Q-Full (`3.164`). These values are DEV evidence and must be replaced by the
single frozen Noisy TEST before submission.

## Four-page ICASSP outline

### Abstract

Use five moves in about 150--180 words:

1. Context: LLMs are increasingly used as generative speech front ends.
2. Failure: TSE evidence may contain the wrong speaker, making conventional
   grounding unsafe.
3. Method: candidate-level confusion repair followed by FSQ code-space
   grounding.
4. Evidence: one clean headline result and one frozen Noisy TEST headline
   result, each including speaker-switch and WER rather than MOS alone.
5. Conclusion: evidence reliability is a prerequisite for trustworthy speech
   LLM grounding.

Do not mention every metric, Pool B, adaptive CSG, or both GNR settings in the
abstract.

### 1. Introduction

- Frame TSE as a speech front end for audio/speech LLMs, not only as a
  conventional source-separation task.
- Give one concrete wrong-speaker failure: the waveform is fluent, but belongs
  to the interferer; perceptual quality cannot detect this error.
- State the gap: existing acoustic grounding generally assumes correct
  evidence and can reinforce a speaker swap.
- State the principle: **repair before grounding**.
- End with exactly three contributions:
  1. identification and paired measurement of wrong-speaker evidence failure;
  2. deployable target-consistent candidate repair plus FSQ code-space
     grounding;
  3. frozen clean/noisy evaluation covering recovery, regression, paired
     significance, SNR robustness, and computation.

Recommended final paragraph sentence:

> Our results show that reliable acoustic evidence is not a by-product of
> grounding but its prerequisite.

### 2. Repair-Before-Grounding LLM-TSE

#### 2.1 Wrong-speaker evidence in LLM-based TSE

- Define mixture `x = s_t + s_i + n`, target enrollment `e_t`, primary
  extraction `y_0`, evidence tokens `z^e`, and LLM output tokens `z`.
- Define the failure: `y_0` is acoustically or linguistically closer to the
  interferer, even when it sounds natural.
- Explain why stronger grounding alone cannot solve this case.

#### 2.2 Target-consistent candidate repair

- Introduce the five Pool-D candidates and their complementary conditioning.
- Give the deployable selector:

  `i* = argmax_i cos(ECAPA(y_i), ECAPA(e_t))`.

- State the no-leakage boundary directly beside the equation.
- Send architecture detail and deterministic-window rules to one compact
  paragraph or supplementary material.

#### 2.3 Code-space grounding of the speech LLM

- Introduce the Q-Full decoder and CosyVoice3 S3/FSQ evidence tokens.
- Give the CSG logit equation and define `lambda` and temporal tolerance.
- Emphasize the matched comparison: identical Pool-D evidence, checkpoint,
  output length, seed, and vocoder; only grounding changes.
- Mention GNR in one short paragraph as an immutable-anchor neighborhood
  ablation. Do not give it equal conceptual weight if it remains below CSG.

Place **Figure 1** across the top or bottom of this section: mixture and
enrollment -> candidate repair -> Pool-D direct / evidence tokens -> Q-Full UD
or CSG -> waveform. Draw oracle and clean references only as dashed evaluation
branches.

### 3. Experimental Protocol

#### 3.1 Data, systems, and frozen evaluation

- Clean natural DEV/TEST and Noisy DEV/TEST; explain the natural plus
  controlled-SNR composition of the 8,400-trial noisy split.
- Six matched systems in the main experiment: Primary, TF-map single expert,
  Pool-D direct, Pool-D -> Q-Full UD, Pool-D -> fixed CSG, and DEV-selected
  GNR.
- Identify Pool B as an efficiency ablation, not another headline system.
- State that all selection occurred on DEV and Noisy TEST ran once from a
  hashed frozen protocol with no post-TEST tuning.

#### 3.2 Reliability, quality, and statistics

- Primary endpoints: raw utterance WER, content-switch rate, acoustic-switch
  rate, and speaker margin.
- Mechanism endpoints: swap recovery, oracle gap, and regression on
  primary-correct controls.
- Secondary endpoints: UTMOS/DNSMOS for perceptual quality, plus eSTOI, LPS,
  and SpeechBERTScore for preservation.
- Use paired bootstrap confidence intervals for continuous metrics and exact
  McNemar tests for binary switch events; report numerator/denominator for all
  switch rates.
- State that DNSMOS and UTMOS do not establish target identity or content
  correctness.

### 4. Results and Analysis

#### 4.1 Candidate repair recovers speaker swaps without harming controls

- Lead with Primary -> Pool-D, not with MOS.
- Report full-set WER and switch reductions, swap recovery, oracle gap, and
  control regression.
- Use the TF-map single expert to prove that per-trial selection adds value
  beyond always replacing Primary with the strongest alternative checkpoint.

**Table 1** should contain two panels, Clean TEST and frozen Noisy TEST, with
the six main systems and columns for WER, content switch, acoustic switch,
speaker margin, and UTMOS. Mark direct and generative rows visually rather
than declaring one universal winner.

#### 4.2 Grounding repaired evidence reduces LLM decoder drift

- Compare Primary-evidence Q-Full with Pool-D-evidence Q-Full to establish the
  value of evidence repair at the LLM interface.
- Compare Pool-D UD with Pool-D fixed CSG to isolate grounding.
- State both benefits and costs. If CSG remains worse than Pool-D direct in
  WER, say so explicitly and describe a reliability--naturalness frontier.
- Treat adaptive CSG and GNR as negative or diagnostic ablations if frozen
  TEST follows DEV.

**Table 2** should combine mechanism and efficiency: candidate availability,
selected recovery, control regression, number of candidates/extractor passes,
LLM and vocoder passes, RTF/latency, and peak memory. This table answers both
“why selection?” and “what does it cost?”.

#### 4.3 Noise robustness and operating-point trade-off

- Show WER and acoustic-switch rate across `-5/0/5/10/15 dB` using the same
  paired trials.
- Report whether CSG beats UD at each SNR, not only in the macro average.
- Discuss the low-SNR failure regime instead of hiding it.

Use **Figure 2** as a compact two-panel per-SNR plot if it remains readable.
If the four-page layout cannot accommodate it, use the speaker-confusion vs.
UTMOS trade-off plot in the main paper and move full per-SNR curves to the
supplementary website.

### 5. Conclusion

Use four or five sentences. Restate the failure, the repair-before-grounding
principle, the clean/noisy confirmation, and the two operating points. End
with the limitation that generative resynthesis still loses content fidelity
relative to the repaired direct waveform; do not add untested future claims.

## Suggested page budget

| Content | Approximate space |
| --- | ---: |
| Abstract + Section 1 | 0.65 page |
| Section 2 + Figure 1 | 1.20 pages |
| Section 3 | 0.55 page |
| Section 4 + tables/figure | 1.40 pages |
| Section 5 | 0.20 page |

Use the optional fifth page only for references. Remove background detail
before shrinking figures or result tables.

## Special-session positioning

The first page must make the speech-LLM connection unavoidable:

- call TSE a **target-conditioned speech front end for a generative speech
  language model**;
- use **acoustic evidence grounding** and **discrete speech-token grounding**
  as key phrases;
- explain that speaker identity is a grounding-validity condition;
- present the direct waveform as the fidelity control and CSG as the LLM
  method, rather than allowing the paper to read as only a multi-expert TSE
  paper.

Suggested keywords:

> target speech extraction; speech language model; acoustic grounding;
> speaker confusion; discrete speech tokens

As of 2026-09-04, the official ICASSP 2027 website exposes the call for
special sessions but not a complete accepted-session list. The exact session
title described by the author could not be verified publicly. Align the final
abstract and keywords to the organizer's invitation once its exact title and
scope are available.

Official constraints used for this plan:

- [ICASSP 2027 Author Guidelines](https://2027.ieeeicassp.org/author-guidelines/)
  specify four pages of technical content and an optional references-only
  fifth page.
- [ICASSP 2027 Call for Papers](https://2027.ieeeicassp.org/call-for-papers/)
  lists Speech and Language Processing, Audio and Acoustic Signal Processing,
  and Machine Learning and Generative AI within scope, with paper submission
  due 16 September 2026.
- [ICASSP 2027 Call for Special Sessions](https://2027.ieeeicassp.org/call-for-special-sessions/)
  states that special-session papers receive a rigorous review similar to
  regular submissions and that acceptance of a session does not guarantee
  acceptance of its papers.

## Submission-critical decision after Noisy TEST

Use the recommended LLM-grounding title only if the frozen Noisy TEST confirms
that fixed CSG improves the matched UD branch on the pre-registered reliability
endpoints without erasing the naturalness benefit over Pool-D direct. If it
does not, retitle and reposition the paper around target-consistent candidate
selection, with LLM grounding as a diagnostic study. This decision uses the
pre-registered criterion; it must not introduce post-TEST tuning.
