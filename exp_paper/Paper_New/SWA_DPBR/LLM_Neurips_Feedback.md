Training-Free Cultural Alignment of Large Language Models via Persona Disagreement
Download PDF
Dao Sy Duy Minh, Trung-Kiet Huynh, Tran Chi Nguyen, Phu Quy Nguyen Lam, Phu-Hoa Pham, Tuan Nguyen, The Anh Han, Long Tran-Thanh
04 May 2026 (modified: 04 May 2026)
NeurIPS 2026 Conference Submission
Conference, Dao Sy Duy Minh, Trung-Kiet Huynh, Tran Chi Nguyen, Phu Quy Nguyen Lam, Phu-Hoa Pham, Tuan Nguyen, The Anh Han, Long Tran-Thanh
Revisions
CC BY 4.0
TL;DR: A training-free, black-box inference-time method that turns within-country persona disagreement into a bounded logit correction, cutting LLM cultural misalignment by 10–24% across 20 countries.
Abstract:
Large language models are increasingly deployed in decisions that require culture-dependent moral judgements, yet they answer as if the whole world thinks with a Western mindset. The Moral Machine experiment showed this is wrong at scale: 40 million judgments across 233 countries reveal that moral preferences are systematically structured by culture, and a model that ignores this variation does not merely underperform, but also imposes one society's intuitions on all others. Existing fixes do not scale to global deployment, as fine-tuning needs per-country preference data and GPU budgets, reward-guided decoding needs per-country reward models, and activation steering needs access to model internals that black-box APIs do not expose. In this work, we focus on this realistic inference-time regime, with no weight updates, no training data, and no internal access. The key observation is that within-country demographic disagreement, not consensus, is the steering signal. When culturally grounded personas agree, the base model is already calibrated. But when they disagree, the spread tells us what to fix and how. We propose DISCA (Disagreement-Informed Steering for Cultural Alignment), which instantiates each country as a panel of four World-Values-Survey-grounded persona agents, converts their disagreement into a bounded, loss-averse correction whose magnitude is set by the panel's variance, and shrinks the correction toward zero when the estimate is unreliable. Across 20 countries and 7 open-weight backbones (2B–70B) from five model families, DISCA reduces cultural misalignment on MultiTP by 10–24% on binary moral dilemmas and 2–7% on open-ended scenarios. Furthermore, a smaller 14B backbone with DISCA reaches lower absolute misalignment than a vanilla 70B model.

Checklist Confirmation: I confirm that I have included a paper checklist in the paper PDF.
Financial Support:  Trung-Kiet Huynh
Reviewer Nomination:  Long Tran-Thanh
Responsible Reviewing: We acknowledge the responsible reviewing obligations as authors.
Primary Area: Socio-technical aspects of AI (e.g., fairness, interpretability, privacy, safety, governance)
Secondary Area: Language and multimodal language models (e.g., text generation, summarization, VQA)
Contribution Type: General: Most submissions will fall into this type.
Academic Integrity: I acknowledge that I have read the NeurIPS Handbook and commit to adhering to all policies in the Handbook (https://neurips.cc/Conferences/2026/MainTrackHandbook), the NeurIPS Code of Conduct and the NeurIPS Academic Integrity Policy.
LLM Usage: Editing (e.g., grammar, spelling, word choice)
Declaration: I confirm that the above information is accurate.
Ready For LLM Feedback: The submitted PDF is ready for LLM feedback.
LLM Feedback Request:  Trung-Kiet Huynh
Submission Number: 18326
Filter by reply type...
Filter by author...
Search keywords...

Sort: Newest First
2 / 2 replies shown
LLM Feedback by Program Chairs
LLM Feedbackby Program Chairs04 May 2026, 02:29Conference, Authors
Feedback:
Hello!

You requested a review of your paper submitted to NeurIPS using the Google Paper Assistant Tool (PAT). The resulting AI Feedback can be found below. Note that this feedback is posted automatically, and is only visible to authors. Importantly, the feedback will not be used in the review process. Reviewers, area chairs, and program committee members will not have access to the PAT feedback.

Disclaimer: Please note that the models used by the PAT pipeline are not infallible; they may hallucinate and make mistakes. Authors should treat the generated feedback with the same critical eye they would apply to a human review.

Summary
The paper introduces DISCA (Disagreement-Informed Steering for Cultural Alignment), a training-free, inference-time method designed to align Large Language Models (LLMs) with diverse cultural moral preferences. By leveraging the World Values Survey (WVS), the authors instantiate four culturally grounded personas per country. A key innovation is treating the variance (disagreement) among these personas as a statistical reliability signal, rather than seeking a simple consensus. DISCA utilizes a loss-averse Prospect-Theory importance sampling mechanism and a dual-pass reliability gate to dynamically scale the magnitude of logit corrections based on persona agreement. The authors evaluate their approach on the MultiTP benchmark across 20 countries, demonstrating that DISCA reduces cultural misalignment by 10-24% and enables a 14B parameter model to outperform a 70B vanilla baseline.

Strengths
Originality in Utilizing Disagreement: The paper presents a highly novel conceptual shift by treating within-country persona disagreement as a statistical reliability signal to dynamically scale the magnitude of logit corrections, moving beyond traditional ensemble methods that treat variance merely as noise to be averaged out.

Scalability and Efficiency: The proposed method is computationally lightweight and operates at inference time without requiring expensive per-country fine-tuning, reward models, or token-heavy multi-agent debate, offering a highly practical solution to the global challenge of Western-centric bias in LLMs.

Strong Empirical Demonstration: The empirical findings are compelling, particularly the result that 'calibration competes with scale,' where a 14B parameter model equipped with DISCA achieves better absolute cultural alignment than a 70B vanilla baseline.

Clear Scope Delineation: The authors provide a valuable scientific contribution by clearly delineating the boundary of their method's effectiveness, demonstrating that persona-disagreement steering succeeds for low-dimensional value alignment (scalar logit gaps) but does not transfer to high-dimensional factual QA tasks.

Weaknesses
Theoretical Claims Regarding James-Stein Shrinkage: The authors might consider revisiting the claim that the method implements a James-Stein shrinkage estimator that 'strictly dominates unshrunk consensus in worst-case MSE.' As the target parameter (the logit gap) is a 1-dimensional scalar, Stein's paradox and strict MSE domination (which require 
 dimensions) do not formally apply. Furthermore, the implemented dual-pass reliability gate is an empirical Monte Carlo variance heuristic rather than a formal oracle shrinkage estimator. Refining these claims to describe the method as a variance-aware heuristic motivated by oracle shrinkage properties would improve theoretical accuracy.

Missing Mathematical Definitions for Reproducibility: Please consider adding explicit mathematical formulations for several core components that are currently missing. Specifically, the gain functions (
 and 
) in Equation 2 require formal definitions (noted as an omitted 
-style gain). Additionally, the blend function in Algorithm 1 is undefined, and the softmax temperature parameter 
 (mentioned in Table 19) is missing from the importance sampling weights in Equation 3. Providing these details would greatly enhance reproducibility.

Clarification of 'Black-Box' Assumptions: The introduction heavily emphasizes operating in a 'black-box' regime. However, the method fundamentally relies on accessing decision-token logits. Since many standard commercial APIs do not expose log probabilities, the authors might consider explicitly clarifying early in the text that the method requires logit-level access, to avoid potentially misleading readers about its applicability to purely text-in/text-out APIs.

Contextualization with Recent Prior Art: To strengthen the paper's positioning, the authors might consider explicitly discussing or comparing against recent concurrent prior art that explores inference-time multi-agent cultural alignment and test-time steering. For instance, comparing with or explicitly discussing the limitations of similar frameworks would be beneficial (Reference Paper: Multiple LLM Agents Debate for Equitable Cultural Alignment, 2025), (Reference Paper: Toward Culturally Aligned LLMs through Ontology-Guided Multi-Agent Reasoning, 2026), and (Reference Paper: Nudging: Inference-time Alignment of LLMs via Guided Decoding, 2024).

Evaluation Split Discrepancies: There appears to be a reporting discrepancy between Table 1 and Table 17. The baselines in Table 1 are described as oracle baselines calibrated on 100% of the human AMCE target, while Table 17 describes them as using a 25% calibration / 75% test split, yet both tables report identical scores (e.g., 0.513 and 0.506). Furthermore, the DISCA mean in Table 17 matches the 100% evaluation score from Table 2. The authors should consider clarifying or correcting these splits to ensure statistical validity.

Potential Issues And Suggestions
[Introduction and Motivation] (Pages: 1-3)
1. Potential Mistakes and Improvements:

Theoretical Claim Alignment: The Introduction states, "We prove (Theorem ??) that the disagreement-driven correction is a James–Stein shrinkage estimator [James and Stein, 1961] that strictly dominates unshrunk consensus in worst-case MSE" (Lines 120–122). However, Proposition 1 (which this appears to refer to) derives an oracle MSE-minimizing scalar shrinkage factor (\gamma^\star) for a fixed, unknown (\Delta_h). It does not appear to contain a proof that the empirical estimator strictly dominates the unshrunk consensus in worst-case MSE. Furthermore, the classical James–Stein estimator typically strictly dominates the maximum likelihood estimator only in dimensions (d \ge 3); for a 1-dimensional logit gap correction, standard empirical scalar shrinkage may not uniformly dominate the unshrunk estimator across the entire parameter space. It is a potential concern that the introductory claim overstates what is formally proven; it may be safer to revise this to reflect what Proposition 1 shows (e.g., that it derives the optimal variance-aware shrinkage factor) rather than claiming James–Stein worst-case domination.

Clarification of "Black-Box" Assumptions: On Lines 54–55 and 80, the method is framed as operating in the "black-box, public-data-only regime," contrasting with methods that need "access to model internals that black-box APIs do not expose" (Lines 9–10). However, the method specifically requires access to the decision-token logits (Line 141), which many standard commercial black-box APIs do not expose (as acknowledged later in Appendix A23). It is a potential concern that the unqualified "black-box" claim in the introduction might be slightly misleading. Briefly clarifying that the black-box assumption specifically requires API log-probability/logit access for the target tokens would make the operational requirements perfectly clear upfront.

2. Minor Corrections and Typos:

Line 120: There is an unresolved reference to "(Theorem ??)," which presumably should point to Proposition 1.

Lines 50–52: There is a sentence fragment caused by the current punctuation: "Reward-guided decoding requires a separate trained reward model per country [...]. While activation steering requires write-access to model internals [...], which black-box APIs do not expose." Consider replacing the period before "While" with a comma or rephrasing to connect the clauses.

[DISCA Methodology and Architecture] (Pages: 3-7, 14-15, 31-38)
Potential Mistakes and Improvements:
Validity of James-Stein Shrinkage Claims: The paper asserts that the disagreement-driven correction is a "James-Stein shrinkage estimator... that strictly dominates unshrunk consensus in worst-case MSE" (Line 120). This claim appears to contain a fundamental statistical misconception:

Dimensionality: Stein's paradox and the strict dominance of the James-Stein estimator over the sample mean only apply when estimating parameters in 
 dimensions. In this context, the parameter being estimated (the logit gap 
 for a single scenario) is a 1-dimensional scalar. In one dimension, the sample mean is an admissible estimator and cannot be strictly dominated. The text in Appendix A8.3 (Line 764) attempts to present a JS shrinkage factor as 
, incorrectly substituting the number of sample agents 
 in place of the parameter dimensionality 
.
Mismatch with Proof and Implementation: Proposition 1 (Appendix A2) mathematically derives the optimal oracle linear shrinkage weight 
, which requires knowing the unknown true parameter 
. It does not establish an operational empirical Bayes estimator. Furthermore, the implemented reliability gate (Eq. 4, Line 255) uses a heuristic 
 based on the Monte Carlo variance of the importance sampling stage, rather than a formal James-Stein formula. The authors should consider removing claims of strict theoretical dominance and recharacterize the method as a variance-aware heuristic motivated by the properties of oracle shrinkage.
Missing Mathematical Definitions (Reproducibility): Several core components required to implement the algorithm independently are conceptually described but lack explicit mathematical formulations:

Gain Functions (
 and 
): Equation 2 relies on the per-cohort gain and the consensus gain. Line 206 states 
 "records how much 
 improves cohort 
's alignment relative to the base prompt," but no mathematical formula is provided to compute it. It is unclear what specific distance metric (e.g., absolute difference or squared error) and reference points are used. Line 1004 refers to an omitted "
-style gain definition in Eq. ??", confirming a formula is missing.
The blend Function: Algorithm 1 computes the final output via \delta_{final} \gets blend(\bar{\delta}, \delta_{base}, \delta^\star) (Line 15). This function is completely undefined in the text. Table 19 mentions an "ESS anchor blend - on", but does not specify the mathematical operation executed when the blend is active.
Importance Sampling Weights: Equation 3 (Line 229) defines the IS weights as 
. However, Table 19 and Appendix A14 (Line 1007) state that a softmax temperature 
 is used, such that 
. Equation 3 should be updated to align with the implementation.
Process Clarity and Pseudocode Ambiguities: There are inconsistencies and omissions within the description of the algorithm steps:

Candidate Perturbation Centering: Line 203 states that candidate perturbations 
 are drawn "in a neighbourhood of the consensus", while Line 939 describes a "Gaussian proposal around 
". It is mathematically ambiguous how the 
 values are evaluated (e.g., whether the evaluated candidate state is structured as 
 or 
).
Undefined Variable 
: Algorithm 1, Line 6 instructs to "Compute 
". The variable 
 is not defined anywhere in the methodology section (it is only briefly defined much later in Appendix A23, Line 1377, as 
).
ESS Guard Fallback Logic: Algorithm 1, Line 11 computes the weighted sum (if ESS > \rho_{eff}). The pseudocode lacks an else branch. Based on the description in Line 1019-1020, it should explicitly indicate the fallback behavior (e.g., 
) when the condition fails.
Minor Corrections and Typos:
Unresolved LaTeX References: The manuscript contains numerous broken reference placeholders (??) that obscure pointers to equations, proofs, and sections.

"Theorem ??" appears frequently (e.g., Lines 120, 764, 775, 1345, 1364) and is presumably intended to refer to Proposition 1.
Algorithm 1, Line 6 refers to [Eqs. ??-??].
Other instances include Section ?? (Line 919), Eq. ?? (Lines 897, 1004, 1356), and (Appendix ??) (Line 1023).
Line 199: "...each is loss-averse which mean a small misalignment..." should be corrected to "...which means a small misalignment...".

Line 204: "...hood of the consensus. Because the consensus gives us only one direction; to find a direction that..." The punctuation should be adjusted (e.g., replacing the semicolon with a comma) for proper sentence flow.

[Theoretical Analysis] (Pages: 4-6, 14-16, 30-31)
1. Potential Mistakes and Improvements
Mathematical Error Regarding James-Stein Dominance (Validity): The paper claims that the disagreement-driven correction is a "James-Stein shrinkage estimator [James and Stein, 1961] that strictly dominates unshrunk consensus in worst-case MSE" (Line 120, referencing "Theorem ??"). Appendix A8.3 (Lines 764-766) further claims the James-Stein factor is 
. This is a mathematical error. Under the model defined in Section 3.2, the target parameter being estimated per scenario (
) is a 1-dimensional scalar. By the admissibility of the sample mean for a 1D Gaussian under squared error loss, no estimator can strictly dominate the sample mean (the unshrunk consensus 
) in worst-case MSE. Stein's phenomenon and the resulting strict dominance require a parameter dimension 
. The formula referenced in the appendix erroneously conflates the number of sampled agents (
) with the parameter dimension (
). Furthermore, Proposition 1 correctly derives an oracle optimal linear shrinkage 
 that depends on the unobservable parameter 
, which is standard bias-variance analysis and not a computable empirical James-Stein estimator. The claims of strict worst-case MSE dominance and the formal James-Stein mathematical connections should be removed or corrected.

Missing Definitions for the PT-IS Utility Function (Clarity): The cooperative utility function 
 in Equation 2 relies on the per-cohort gain 
 and the consensus gain 
. The text provides a conceptual description that 
 "records how much 
 improves cohort 
's alignment relative to the base prompt" (Lines 206-207), but no formal mathematical definition is provided for how these gains are computed from the candidate perturbation 
, the consensus 
, the base logit gap 
, and the agent's gap 
. Appendix A14 (Line 1004) references "the 
-style gain definition in Eq. ??", confirming an accidental omission. Without exact formulas, the PT-IS objective landscape cannot be formally understood or reliably reproduced.

Theoretical Disconnect in the Reliability Gate (Validity): Section 3.3.2 implements a reliability gate 
 based on the variance between two stochastic importance sampling passes, 
. Line 251 asserts this is the "operational analogue" of the within-cohort agent variance 
 from Proposition 1. However, 
 captures algorithmic numerical integration instability (Monte Carlo sampling noise), while 
 captures true demographic preference disagreement (
). Shrinking based on computational sampling noise is conceptually distinct from the optimal shrinkage derived in Proposition 1 based on population variance. The theoretical justification should explicitly distinguish the practical Monte Carlo variance heuristic from the formal derivation in Section 3.2, and the claim on Lines 65-66 that the attenuation "is not heuristic: it implements a variance-aware shrinkage estimator whose shrinkage weight is determined by the panel's empirical disagreement" should be revised to reflect the actual implementation.

Undefined blend Function and Contradictory Shrinkage Target (Clarity): Algorithm 1 (Line 15) outlines the final inference step as applying the function 
. This function is mathematically undefined. This omission creates a theoretical ambiguity: Section 3.3.2 (Line 259) claims that as the reliability weight 
, "the correction contracts toward the no-op direction" (i.e., 
). However, if the perturbations 
 are zero-centered, the IS offset 
 merely represents an adjustment. If the blend function simply adds this offset to the consensus 
, then as 
, the final gap reverts to the unshrunk consensus 
, not the no-op 
. The mathematical formulation of blend must be explicitly provided to verify that it achieves the theoretically claimed shrinkage behavior.

Missing Softmax Temperature Parameter in Eq. 3 (Correctness): In Equation 3 (Line 229), the importance sampling weights are defined as 
. However, Appendix A14 (Line 1007) and the hyperparameter table (Table 19) state that these weights are computed using an IS softmax temperature parameter 
, such that 
. Incorporating 
 into Equation 3 is necessary to ensure mathematical consistency with the implementation and the KL-regularised Boltzmann posterior grounding.

2. Minor Corrections and Typos
Incorrect Equation Reference: Appendix A13 (Line 939) states, "This is what licences us to read our Eq. 2 as an inference step rather than an arbitrary scalar interpolation." Equation 2 defines the deterministic utility function 
. Equation 3 defines the actual softmax-weighted aggregation step (path-integral/importance-sampling). The reference should point to Equation 3.

Bounds in Proposition 1: The statement of Proposition 1 (Line 174) claims 
 is bounded in the open interval 
. As correctly noted in the proof (Line 645), this should be the closed interval $$, since 
 when 
, and 
 when 
.

Notation in Estimator Definition: In Section 3.2 (Line 168), the scaled adjustment is denoted as 
. Notationally, using 
 or 
 would be strictly consistent with the variable 
 on the right-hand side and in the 
 definition.

Algorithm 1 Variables: In Algorithm 1, Line 6 instructs to "Compute 
". The variable 
 is mathematically undefined and unused in the rest of the algorithm. Also, Line 14 assigns 
, but Line 15 passes 
 to the blend function. The variable naming should be unified.

Sentence Fragment: Line 166 ("Where 
 is the logit gap produced by the unconditioned base prompt...") is a sentence fragment and should be combined with the preceding sentence.

Broken References: There are multiple unresolved LaTeX cross-references rendered as "??" throughout the theoretical sections, including "Theorem ??" (Lines 120, 764, 775, 786, 1345, 1364, 1483), "(PT-IS; §??)" (Figure 1 caption, Line 149), "[Eqs. ??–??]" (Algorithm 1, Line 6), "Eq. ??" (Lines 897, 1004, 1356), and "(Appendix ??)" (Lines 955, 1023).

[Empirical Evaluation and Ablations] (Pages: 6-10, 16-30, 38-42)
1. Potential Mistakes and Improvements
Inconsistent Evaluation Splits (Table 1 vs. Table 17): There appears to be a reporting discrepancy regarding the baseline evaluation splits. Table 1 presents Temp. scaling and Margin scaling as oracle baselines (calibrated on 100% of the human AMCE target) with MIS scores of 0.513 and 0.506. Table 17 presents inference-time baselines Tc scale and Margin mc with the exact same scores (0.513 and 0.506). However, Appendix A10 (Line 829) states that Table 17's baselines "use a 25% per-country calibration split and are evaluated on the remaining 75% test split." Furthermore, Table 17 reports the DISCA mean as 0.346, which is identically the 100% evaluation score from Table 2. It is statistically improbable that the 75% split and 100% split yield identical 3-decimal-place averages, suggesting the evaluation sets are mismatched or results were inadvertently copied over.

Missing LLM Judge Specification for Open-Ended Benchmark: Section 3.4 and Appendix A3.1 introduce an open-ended ethical evaluation that relies on a "separate large language model (LLM) judge" to extract pseudo-logit gaps from free-form responses. The identity, parameter size, and prompt configuration of this judge model are not documented anywhere in the paper. Additionally, the acronym "SAFE" in the Table 7 caption is undefined. Providing these details is necessary for the reproducibility of the open-ended benchmark results.

Misstated Baseline Performance in Figure 4 Caption: The caption for Figure 4 states that the 14B Phi-4 model reaches an MIS of 0.346, "lower than a 70B model without DISCA (Llama-3.3-70B, 0.668)." However, according to Table 2 and Table 6, 0.668 is the MIS for Llama-3.3-70B with DISCA (the vanilla MIS without DISCA is 0.849). The caption misstates the baseline performance by attributing the post-intervention score to the vanilla model.

Activation Steering Baseline Mismatch: Appendix A19.1 (Line 1289) states that for the activation steering baseline, "Vectors are extracted from layer 32 (transformer midpoint) of Llama-3.1-70B and applied at inference time". However, Table 1 evaluates the Activation Steering baseline on the Phi-4 (14B) model. A vector extracted from a 70B model cannot be directly added to a 14B model's residual stream due to dimension mismatches. The text should clarify if vector extraction was actually performed per-model for this evaluation.

Scale Discrepancy for AMCE/MIS: Appendix A16 (Line 1241) states that ground-truth AMCEs are converted to a [0, 100] scale, and that "Model AMCEs are computed on the same scale." Figure 3 also reports 
 distances on this scale (e.g., 83.3 to 64.6). However, the MIS values reported in all tables (e.g., Tables 1, 2, 4, 6) range from roughly 0.2 to 1.0. The text should clarify whether the tables report the metric scaled down (e.g., by a factor of 100) to resolve this apparent contradiction.

Missing Mathematical Definitions (Eq. 2 & Algorithm 1): Several components of the PT-IS stage are not mathematically defined, hindering exact reimplementation. Eq. 2 relies on gain functions 
 and 
, but their exact formulation is missing. Additionally, Algorithm 1 (Line 15) uses a blend function, and Table 19 refers to an "ESS anchor blend," but the mathematical operations behind these functions are not defined.

Equation 3 vs. Table 19 Temperature Parameter: Equation 3 defines the IS weights as 
. However, Table 19 introduces an "IS softmax temp. 
", and Line 1007 states 
. Equation 3 should be updated to include the 
 parameter.

Ambiguous Ablation Footnote: Footnote 1 (Page 8) states "dual-pass reliability disabled for clean isolation" for the Table 4 ablations. However, the "Full DISCA" configuration in Table 4 reports the exact same MIS (0.362 for Qwen2.5-7B) as the main results in Table 2, which include the dual-pass gate. If the gate were disabled for the entire table, "Full DISCA" should theoretically match "Always-on PT-IS" (0.379). The footnote should clarify that this condition likely applies only to specific ablation rows.

Overstated Theoretical Claim: Line 120 claims to "prove (Theorem ??) that the disagreement-driven correction is a James-Stein shrinkage estimator that strictly dominates unshrunk consensus in worst-case MSE." However, Proposition 1 (Appendix A2) only derives a scalar MSE-minimizing shrinkage factor. This does not constitute a formal proof of strict MSE domination for a multi-dimensional James-Stein estimator (which mathematically requires dimension 
). The language should be adjusted to accurately reflect what is proven in Proposition 1.

Missing Model Attribution in Table 15: Table 15 summarizes countries with negative Pearson 
 after DISCA but does not specify which backbone models these rows correspond to (e.g., the BRA 
MIS of -9.3% matches Phi-4 in Table 6, while ARG +12.2% matches Llama-3.3-70B). The table should clearly map each row to the specific model evaluated.

Missing Implementation Details for Table 1 Baselines: Table 1 lists several training-free baselines ("PRISM-Style Prompt", "Fixed logit offset", "WVS Profile Prompt"), but their specific implementations are not detailed in Appendix A19 or the main text.

2. Minor Corrections and Typos
Unresolved Cross-References: There are numerous unresolved LaTeX references (rendering as ??) throughout the text. Examples include Line 120 ("Theorem ??"), Line 235 ("§??"), Algorithm 1 ("Eqs. ??–??"), Line 652 ("§??"), Line 764 ("Theorem ??"), Line 897 ("Eq. ??"), Line 919 ("Section ??"), Line 955 ("Appendix ??"), and Line 1023 ("Appendix ??").

Table 6 Notation: The columns in Table 6 are labelled MISv and MISs. While the legend clarifies this means "vanilla vs. DISCA," defining MISs more explicitly in the table header (e.g., as "steered") would improve readability.

Em-dash Formatting: Throughout the text, a single unspaced hyphen is frequently used in place of an em-dash or spaced en-dash, causing words to run together (e.g., Line 664: "Utilitarianism-they", Line 670: "balance-not raw scale-explains", Line 673: "Indonesia-with", Line 708: "countries-predominantly").

Algorithm 1 Variable Mismatch: In Algorithm 1, Line 14 assigns the value to \delta^\star_{IS}, but Line 15 passes \delta^\star to the blend function. This should be made consistent.

Undefined Terms in Table 18: Table 18 uses acronyms in the "Test" column (e.g., CS-Clamp, ARGS-WVS, ARGS-Unif, ARGS-PT) without defining them in the caption or text.

[Conclusion, Impacts, and Compliance] (Pages: 9-13, 42-51)
1. Potential Mistakes and Improvements:

Inconsistent Performance Claim: In the Conclusion (Line 400), the paper claims that DISCA reduces cultural misalignment by "10–24% on binary dilemmas". Earlier in the abstract, this exact 10–24% claim is explicitly applied across the "7 open-weight backbones (2B–70B)". However, Table 2 reports that the 2B model (Gemma-4-E2B) achieves a 3.4% gain. It appears the 10–24% range excludes the 2B model, making the explicit inclusion of the "(2B–70B)" size range slightly misleading. Consider updating the stated range to accurately reflect all 7 models (e.g., 3–24%), or clarifying that the 10–24% range applies specifically to the models 
 3.8B.

Broken Cross-References: There are several unresolved LaTeX references (??) within the reviewed sections that obscure pointers to key methodological and theoretical details. These should be corrected:

In Appendix A22 (Lines 1345 and 1364) and the NeurIPS Paper Checklist (Question 3, Line 1483), the text refers to "Theorem ??". Based on the main text, this is likely intended to reference "Proposition 1".
In Appendix A22 (Line 1356), there is a reference to "Eq. ??".
In the NeurIPS Paper Checklist (Question 4, Line 1505), the text references "Eqs. 2–??".
In the NeurIPS Paper Checklist (Question 5, Line 1556), the text refers to datasets cited in "§??".
In the NeurIPS Paper Checklist (Question 16, Line 1831), there is an unresolved reference to "§??".
2. Minor Corrections and Typos:

Line 556: The reference for Qiu et al. (2025) is missing the publication venue (e.g., "arXiv preprint").