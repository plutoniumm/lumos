# Related work

This survey positions the measurements in this repository against published
work. Each section states what the literature reports, then what this project
measured, then whether they agree. Citations were checked to resolve; anything
unverified is marked.

---

## 1. Activation steering

The line begins with editing internal activations rather than weights.

**Representation Engineering** ([Zou et al., 2023](https://arxiv.org/abs/2310.01405))
formalises reading from and writing to a model's internal representations as a
control method, and demonstrates behavioural control across safety-relevant
dimensions.

**Activation Addition** ([Turner et al., 2023](https://arxiv.org/abs/2308.10248))
shifts inference-time activations along a direction obtained from a contrastive
prompt pair. It needs as few as 2 samples and applies the vector to a subset of
sequence positions.

**Inference-Time Intervention** ([Li et al., 2023](https://arxiv.org/abs/2306.03341))
trains linear probes to find attention heads whose activation distributions
differ for true and false statements, then steers those heads toward truthful
outputs. It requires labelled data and dozens of samples.

**Contrastive Activation Addition** ([Rimsky et al., ACL 2024](https://aclanthology.org/2024.acl-long.828/),
[arXiv 2312.06681](https://arxiv.org/abs/2312.06681)) averages the difference in
residual stream activations between positive and negative examples of a
behaviour, then adds the result at all token positions after the prompt with a
positive or negative coefficient. Evaluated on Llama 2 Chat, it works on top of
finetuning and system prompts and reduces capabilities minimally.

**This project** uses the CAA formulation, `v = E[h+] - E[h-]`, as one of three
estimators, and confirms the sign property Rimsky et al. rely on: at
`alpha = -1.0` the structural vector at layer 17 lengthens sentences by 60%
(own-axis effect -0.598). **Agrees.**

The novel element here is not the recipe but the question. Prior work extracts
one direction for one behaviour. This project asks whether two *components* of a
single behaviour are separable from each other, which requires measuring
inter-vector cosine rather than steering efficacy. That measurement does not
appear in the works above.

---

## 2. Instruction-following steering

**Improving Instruction-Following through Activation Steering**
([Stolfo et al., ICLR 2025](https://openreview.net/forum?id=wozhdnRCtw),
[arXiv 2410.12877](https://arxiv.org/abs/2410.12877),
[code](https://github.com/microsoft/llm-steer-instruct)) computes a steering
vector as the difference in residual stream activations between inputs **with and
without** an instruction. It searches for the best layer, evaluates on IFEval
with Phi-3.5, Gemma and Mistral, and steers constraints on output format, length
and word inclusion. It also reports that steering reinforces adherence even when
the instruction is present, countering instruction drift.

**This project** adopts that recipe and reports a limitation of the
with-versus-without contrast that the source work has no reason to encounter.
Differencing against a no-instruction prompt lets prompt length into the delta,
and that component is common to both axes. Measured on the same 256 texts, the
no-instruction contrast gives mean absolute cosine **0.542**; an axis-matched
placebo of comparable length reduces this to **0.223**.

This is an **extension, not a contradiction**. Stolfo et al. extract one vector
per instruction and never require two vectors to be distinguishable from each
other, so a shared length component is invisible in their setting: it is common
to the vector and its own baseline. The confound only becomes measurable when two
instruction-derived vectors are compared. Anyone extracting multiple
instruction vectors intended to be independent should measure their mutual cosine
before assuming they are.

---

## 3. Frameworks and strength parameterisation

**EasySteer** ([arXiv 2509.25175](https://arxiv.org/abs/2509.25175)) unifies
steering behind `f(h) = h + alpha*v` with extraction by CAA, PCA variants, linear
probing and sparse autoencoder features, and supports layer, position, token and
stage granularity. Related tooling includes `repeng`, `pyreft`, `EasyEdit2`,
LM-Steer and LoReFT.

Across the works surveyed, `alpha` is a **raw scalar** applied identically at any
layer. No surveyed work normalises it by the layer's activation magnitude.

**This project** measures residual norms growing from **1.60 at layer 0 to 52.04
at layer 27**, a factor of 33, and therefore expresses `alpha` as a fraction of
the layer's mean activation norm: `alpha * unit(v) * mean_norm[layer]`. Without
this a layer sweep compares interventions differing 33-fold in magnitude, and any
apparent depth profile is partly an artifact of normalisation.

**Untested in the literature as far as this survey found.** This is the clearest
methodological contribution here. It is a small change and it is the precondition
for the depth measurement in section 5 meaning anything.

---

## 4. Steering failure modes

**Understanding (Un)Reliability of Steering Vectors**
([Braun et al., 2025](https://arxiv.org/abs/2505.22637)) reports that steering
vectors trained on one prompt set often fail to generalise to another, that some
behaviours are substantially more steerable than others, that failures are
systematic rather than random, that effectiveness depends strongly on which layer
is targeted, and that **coefficient sensitivity is non-linear**, so modest
changes in strength can alter results sharply.

**This project agrees on every point and quantifies two of them.**

Non-linearity: structural own-axis effect at layer 17 peaks at **+0.339** at
`alpha = 1.0` and collapses to **+0.027** at `alpha = 1.25`. Response is
therefore not monotone in strength, which means any bisection search over
`alpha` must be confined to a bracket verified monotone and gate-passing. A naive
bisection enters the degenerate region and reports it as success, because broken
text scores as simple.

Differential steerability: the structural axis reaches own-axis effect
**+0.339** while the lexical axis reaches **+0.155** at its best gate-passing
setting, on the same model with the same estimator. One component of one
behaviour is roughly twice as steerable as the other.

Layer dependence: structural response is confined to layers **10 to 18** of 28
and is absent outside that band.

**Degeneration.** [Holtzman et al. (ICLR 2020)](https://arxiv.org/abs/1904.09751)
established that maximisation-based decoding produces text that is bland,
incoherent, or stuck in repetitive loops, and that such text differs from human
text in likelihood, diversity and repetition. This project observes the same
collapse induced by steering strength rather than by decoding: at `alpha = 2.0`,
75% of outputs degenerate.

The corollary that follows from Holtzman et al. and that this project measured
directly: **repetition is low-entropy, so degenerate loops have low perplexity**.
The broken fragment `organe-cepted control-cepted control-cepted contr` scored
perplexity **18.8** while the valid text *The cell makes energy. The cell uses
energy. The cell needs energy.* scored **37.0**. Perplexity therefore cannot
serve as a degeneration filter for loops, only for high-entropy breakage.
**Consistent with** the likelihood analysis in Holtzman et al., though stated
there as a property of human-versus-machine text rather than as a filter design
constraint.

---

## 5. Text simplification: task and metrics

**SARI** ([Xu et al., TACL 2016](https://aclanthology.org/Q16-1029/)) scores
n-grams added, kept and deleted against the source and multiple references. It is
the standard metric. Its known weakness is that it measures n-gram overlap and
penalises simplifications that are synonymous with a reference but worded
differently.

**ASSET** ([Alva-Manchego et al., ACL 2020](https://aclanthology.org/2020.acl-main.424/))
supplies 10 references per source over 2,359 sentences reused from TurkCorpus,
with references featuring lexical paraphrase, splitting, deletion and reordering.

**OneStopEnglish** ([Vajjala and Lucic, 2018](https://aclanthology.org/W18-0535/))
provides articles at three reading levels.

**This project** reports SARI 45.44 for no intervention, 45.87 for structural
steering and 40.40 for prompt-only on ASSET test at n=32.

It also encountered the SARI weakness above concretely. The largest single SARI
loss for steering, item 20 at **-15.4**, has faithfulness **0.986** and reads as a
competent shortening: *He was named Sportsman of the Year in 1982 by Sports
Illustrated.* SARI penalised it for differing from all ten references.
**Agrees with** the documented limitation.

**Data warning not found in the literature.** The flat three-class
OneStopEnglish distributions are not index-aligned despite appearing so.
Content-overlap matching identified the presumed partner for only 184, 186 and
187 of 189 pairs across the three level combinations, and index 53 pairs a
skydiving article with a royal birth. This project used OneStopQA instead, which
carries explicit title, paragraph and level keys. Anyone zipping the three
class-label splits to obtain aligned triples will produce silently mismatched
pairs.

---

## 6. Flesch-Kincaid as a metric: direct convergence

**[Tanprasert and Kauchak, GEM 2021](https://aclanthology.org/2021.gem-1.1/)**,
"Flesch-Kincaid is Not a Text Simplification Evaluation Metric", show that "the
FKGL score can easily be manipulated to improve the score dramatically with only
minor impact on other automated metrics (BLEU and SARI)", and conclude FKGL
should not be used as an evaluation metric for simplification, recommending its
component statistics for post-hoc analysis instead. The broader survey by
[Alva-Manchego et al. (Computational Linguistics, 2021)](https://doi.org/10.1162/coli_a_00418)
finds Flesch-based metrics have the lowest correlations with human judgement,
and are unsuitable for sentence-level evaluation.

**This project reached the same conclusion independently, by a different route,
and the agreement is exact.** A faithful 17-word paraphrase with NLI 0.990
against its source scored FKGL **14.3**, while a 60-word rewrite that invented
three entities and scored NLI 0.090 scored FKGL **7.0**. The mechanism is
visible in the formula: `0.39(W/S) + 11.8(Y/W) - 15.59` has `S = 1` at
sentence length, so the first term is the sentence's own word count and the
metric rewards padding.

An early comparison in this project used FKGL to conclude prompting outperformed
steering. That conclusion was wrong and was retracted. Tanprasert and Kauchak
predicted exactly this failure four years earlier. **Strong agreement**, and a
reminder that the relevant literature should be read before the metric is
trusted, not after it misleads.

---

## 7. LLM simplification and meaning preservation

**[Agrawal and Carpuat, TACL 2024](https://arxiv.org/abs/2312.10126)**, "Do Text
Simplification Systems Preserve Meaning? A Human Evaluation via Reading
Comprehension", evaluate meaning preservation by asking comprehension questions
about simplified text. Their finding: even the best-performing supervised system
leaves at least **14%** of questions unanswerable from the simplified content.
All systems showed measurable information loss.

**[BLESS](https://arxiv.org/abs/2310.15773)** benchmarks LLMs on sentence
simplification, and error-based human assessment of LLM simplification
([arXiv 2403.04963](https://arxiv.org/abs/2403.04963)) catalogues the error types
that result.

**This project agrees and adds a magnitude for the prompt-only condition.**
Prompt-only simplification scored bidirectional-NLI faithfulness **0.431**, with
only 40.6% of outputs passing a 0.50 gate, and was **significantly worse than no
intervention at all** on SARI (-5.04, 95% CI -8.13 to -1.96). Inspection shows
the mechanism: asked to simplify a sentence about income concentration, the
model produced a cookie-jar analogy and dropped the 6.37% figure (NLI 0.025); on
another item it invented universities, books and domain experts absent from the
source (NLI 0.090).

The contribution here is the **controlled comparison**. Agrawal and Carpuat
measure that simplification systems lose meaning. This project measures that at
matched conditions steering loses **less** of it than prompting: paired per item,
faithfulness +0.36 (CI +0.23 to +0.50, 27 wins to 5).

---

## 8. NLI as a faithfulness metric

**SummaC** ([Laban et al., TACL 2022](https://aclanthology.org/2022.tacl-1.10/))
revisits NLI models for inconsistency detection by segmenting documents into
sentence units and aggregating pairwise scores, rather than scoring whole
documents at once. **AlignScore** ([Zha et al., ACL 2023](https://aclanthology.org/2023.acl-long.634/))
trains a unified alignment function over 4.7M examples spanning 7 tasks.

Both define consistency as each output sentence being supported by, and not
contradicted by, the source.

**This project** uses `roberta-large-mnli` with **sentence-level aggregation**,
matching SummaC's finding that sentence segmentation is what makes NLI work here,
and adds a **bidirectional** requirement: source-entails-output alone rewards
deleting facts, since a shorter claim is easier to entail, and
output-entails-source alone rewards inventing them. Requiring both bounds
omission and invention at once.

The known weakness, that NLI penalises heavy paraphrase, is acknowledged rather
than resolved. This project checked it by inspection on the cases driving its
conclusions: the low prompt-only scores correspond to genuinely dropped figures
and invented entities, not to stylistic distance. AlignScore or SummaC would be
the stronger instrument and were not used.

---

## 9. Controllable simplification without steering

**ACCESS** ([Martin et al., 2020](https://arxiv.org/abs/1910.02677)) conditions a
seq2seq model on control tokens for four attributes: character length ratio,
replace-only Levenshtein similarity, aggregated word frequency ratio, and
dependency tree depth ratio. Length and lexical complexity become adjustable at
inference. **MUSS** ([Martin et al., LREC 2022](https://aclanthology.org/2022.lrec-1.176/))
extends this to multilingual unsupervised training on mined paraphrases.

This is the closest competing approach on the axis this project cares about, and
it is worth stating plainly that **ACCESS already offers separable structural and
lexical control**, via its length ratio and word frequency ratio tokens, with
better range than steering achieves here.

The differences are the trade. Control tokens require training a conditioned
model; steering requires none and runs on a stock instruction-tuned model.
Control tokens are discrete values seen during training; `alpha` is continuous.
Against that, ACCESS controls are learned to be effective, whereas steering is
limited to directions the model already represents, which is what bounds this
project's range: no target reading level in `config.TARGET_FKGL` was reachable,
with gate-passing grade spanning only 12.19 to 14.76 against targets of 4, 6 and
8.

**For the product goal, ACCESS-style control tokens are the stronger method.** Steering's advantage is that it needs no training run.

---

## 10. Summary

| Finding here | Closest published work | Relation |
|---|---|---|
| Negative `alpha` reverses the effect | Rimsky et al. 2024 | agrees |
| Instruction-difference recipe works | Stolfo et al. ICLR 2025 | agrees |
| No-instruction contrast gives cosine 0.542 between two axes; placebo gives 0.223 | Stolfo et al. do not compare two vectors | extends |
| `alpha` normalised per layer by activation norm (33x range) | not found | untested |
| Structural response confined to layers 10-18 | Braun et al. 2025 report layer dependence | agrees, quantified |
| Response non-monotone in `alpha` (+0.339 at 1.0, +0.027 at 1.25) | Braun et al. 2025 report coefficient sensitivity | agrees, quantified |
| Structural 2x more steerable than lexical | Braun et al. 2025 report differential steerability | agrees, quantified |
| Loops have low perplexity (18.8 vs 37.0), so perplexity cannot filter them | follows from Holtzman et al. 2020 | consistent, restated as a design constraint |
| FKGL invalid at sentence length; rewards padding | Tanprasert and Kauchak 2021 | **strong agreement, independently rediscovered** |
| Prompt-only simplification loses meaning (NLI 0.431, SARI -5.04 vs no intervention) | Agrawal and Carpuat 2024 (14% unanswerable) | agrees, different instrument |
| Sentence-level NLI aggregation needed | Laban et al. 2022 | agrees |
| Bidirectional entailment requirement | SummaC and AlignScore are unidirectional support checks | extends |
| CAA conflates the axes (0.673) because both outputs share "simpler than source" | not found | untested |
| Two simplification axes separable by depth | not found for simplification | untested |

### Nothing here contradicts published work

The closest thing to a contradiction is section 2, and on inspection it is an
extension: the confound this project found is invisible in the setting where the
recipe was introduced, because that setting never compares two
instruction-derived vectors to each other.

The most useful convergence is section 6. FKGL misled this project exactly as
Tanprasert and Kauchak (2021) said it would, and the error was found by
measurement rather than by reading. The literature had the answer first.

### Gaps in the surveyed literature

Three claims in this project could not be checked against prior work:
per-layer normalisation of steering strength, the depth separation of two
components of one behaviour, and the mechanism by which output-contrastive
estimation conflates sub-behaviours. These are stated as untested rather than as
novel, since absence from a survey of this size is weak evidence.
