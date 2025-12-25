Summary
This paper tackles automated program repair in low-resource programming ecosystems where tests, training corpora, and mature toolchains are lacking. It proposes a static-oracle-guided LLM repair paradigm that separates where-to-fix (a precision-first static analyzer used as an iterative oracle) from how-to-fix (LLM synthesis augmented by retrieval of curated defect-repair pairs). The authors instantiate this paradigm for ArkTS/OpenHarmony in a system called HapRepair, integrating a new static analyzer (HomeCheck), static analysis–driven context extraction/combination, and RAG with a curated knowledge base; on 35 projects (8,664 findings), HapRepair reduces static-oracle findings by ~97% in five iterations and offers ablations to explain the contributions of retrieval, context scoping, and diff scaffolding.

Strengths
Technical novelty and innovation
Treating a language-specific static analyzer as a precision-first oracle for both localization and iterative verification is a clean, reusable design principle for test-scarce settings; the paper distills four oracle properties (precision-first, stable rule IDs, structured localization, repair-oriented context boundaries) that are broadly transferable.
The Surrounding Context Extractor and Context Combination Tool are pragmatic, analysis-backed mechanisms to bound and merge repair scopes in multi-defect files, addressing a real, often overlooked failure mode of LLMs in long, noisy contexts.
Building a focused RAG knowledge base of 426 defect-repair pairs for ArkTS performance/security rules is a useful bootstrapping resource for a low-resource language.
Experimental rigor and validation
End-to-end evaluation on 35 real projects with 8,664 findings under a frozen oracle, five iterations, and multiple LLMs provides a convincing scale demonstration for the specific ecosystem.
Ablations isolate the roles of RAG quantity (Top-1/Top-3/Top-5), context strategy (surrounding vs. full file), and diff scaffolding, aligning with and explaining the observed gains.
A stratified human-expert semantic check complements static-oracle compliance, with inter-rater agreement statistics reported.
Clarity of presentation
The where-to-fix/how-to-fix separation and the role of the static oracle are clearly articulated; threats to validity are discussed with appropriate caveats about “guideline compliance” versus full semantic correctness.
The system architecture (HomeCheck) and workflow are described at a level sufficient to understand the moving parts and how they interact.
Significance of contributions
The paper addresses an increasingly common and important practical setting (emerging languages/platforms without tests or large corpora), and the proposed design offers a path forward when test-based or data-hungry APR is impractical.
The distilled oracle properties and context-control techniques have utility beyond ArkTS/OpenHarmony.
Weaknesses
Technical limitations or concerns
The main success metric is reduction in static-oracle findings under a frozen rule set; while the authors appropriately caveat this as guideline compliance, the lack of stronger semantic validation (beyond a 150-file sample) limits claims about correctness and safety of repairs at scale.
The precision of HomeCheck is estimated from a 200-finding sample (0 FPs), with no recall measurement; the small sample and absence of recall or CIs make the “precision-first” claim somewhat fragile.
The approach implicitly assumes availability of a high-precision static oracle; external validity to ecosystems with weaker oracles is discussed but not empirically evaluated.
Experimental gaps or methodological issues
Missing baselines that would strengthen attribution: (i) template-only/deterministic auto-fixers where applicable, (ii) LLM-only repair prompted by rule descriptions without the oracle-driven context extraction, and (iii) comparisons to prior static-oracle–guided APR pipelines (e.g., Infer-guided systems, compilation-oracle CI repair).
Cost and efficiency analysis lacks detail for the LLM portion (token/latency per patch, per iteration), though scan times are reported; without this, the “cost-controllable” claim is hard to assess.
Per-rule and per-change-type breakdowns are limited; more granular analyses would help understand where the method struggles (e.g., lifecycle/resource rules vs. small syntactic tweaks).
Clarity or presentation issues
Several tables and snippets show extraction artifacts and truncations (e.g., broken table entries), which impede precise interpretation in places (though not fatal to overall understanding).
Algorithmic details for context extraction/combination are high-level; for reproducibility, more specifics (e.g., definition of overlap/merge, handling of cross-file dependencies) would be helpful.
Missing related work or comparisons
The paper would benefit from a deeper positioning against static-oracle/procedural APR frameworks and feedback-driven repair benchmarks (e.g., compiler/static-linter guided repair in Shadow Job; broader observations in FeedbackEval; and taxonomy/systems in recent LLM-APR surveys). Explicit empirical or conceptual comparisons would clarify novelty boundaries and trade-offs.
Detailed Comments
Technical soundness evaluation
The decomposition into a precision-first oracle for where-to-fix and an LLM for how-to-fix is sound and well-motivated by the preliminary localization study. The emphasis on structured outputs and stable identifiers is practically important and often overlooked.
The RAG + context extractor combination addresses two central low-resource challenges: knowledge scarcity and attention noise. The ablations (Top-1/Top-3/Top-5; surrounding vs. full-file) empirically support the design and reflect known LLM behavior in long contexts.
Verification remains the soft spot. Syntax/declaration checks + re-running the same static oracle can green-light patches that shift, suppress, or overfit to the oracle; the lightweight CFG guardrail is sensible but weak. The human study partly mitigates this, but broader semantics or downstream CI/test signals would strengthen the conclusions.
Experimental evaluation assessment
Scope/scale: Evaluating on 35 projects and 8,664 findings is substantial for a young language. The five-iteration curves show steady convergence, and the cross-model comparison suggests the pipeline is robust to model choice.
Metrics: Reporting “remaining findings” and “guideline-compliance repair rate” is appropriate under the chosen oracle; however, additional metrics such as patch size distributions, proportion of add/modify/delete operations, and rollback rates would provide safety/composure insights.
Oracle precision: The 0-FP sample is encouraging, but please add statistical intervals (e.g., Clopper–Pearson lower bound for zero failures) and per-rule precision breakdowns; absence of recall leaves open whether the tool primarily fixes “easy-to-detect” issues.
Human validation: The stratified file-level assessment (n=150) with inter-rater agreement is commendable; more detail on stratum definitions/weights, and examples of the 2% regressions, would be valuable.
Comparison with related work (using the summaries provided)
Shadow Job (2510.13575) shows compiler-oracle–guided LLM fixes for CI build failures. HapRepair’s static-oracle guidance is similar in spirit (non-test oracles that are scalable) but targets guideline-level rule violations in a low-resource language and adds RAG/context extraction. A direct baseline using a compiler/linter-only oracle (without HomeCheck’s ArkTS-specific analyses) would clarify HomeCheck’s added value.
FeedbackEval (2504.06939) demonstrates that structured feedback (tests/linters) and iterative loops matter, with diminishing returns after 2–3 iterations—mirrored here by strong early rounds. Positioning HapRepair’s ablations alongside these findings would strengthen the generalizability argument.
Surveys (2301.03270; 2405.01466; 2506.23749) situate static-oracle–guided procedural pipelines as a recognized category in LLM-APR. The paper’s design principles align with the “analysis-augmented generation” and RAG observations in these surveys, but an explicit comparison to representative systems (e.g., Infer-guided repairs or D4C-like pipelines) would better highlight novelty.
CompDefect (2204.04856) focuses on template-friendly, single-statement repairs; the authors’ “template-sufficient vs LLM-needed” discussion is apt and could be strengthened with a template-only baseline on clearly templateable rules.
Discussion of broader impact and significance
The work offers a practical recipe for ecosystems with scarce tests and data: use a precise, structured static oracle to bound and verify; add RAG to bridge knowledge gaps; and enforce context discipline to reduce attention noise. If HomeCheck/HapRepair are indeed open-sourced, this could materially accelerate quality improvements for ArkTS and inspire analogous efforts in other niche languages.
Risks include over-reliance on the oracle (compliance gaming), silent regressions that evade static checks, and the potential for LLM patches to homogenize code toward the examples in the knowledge base. The inclusion of rollback and human validation is a step in the right direction; future work could integrate selective dynamic checks or lightweight runtime probes where feasible.
Questions for Authors
Can you provide a stronger set of baselines: (a) template-only/deterministic auto-fixers for rules that admit them, (b) LLM-only repair without the oracle-driven context extractor (e.g., rule description + full file), and (c) an existing static-oracle pipeline (e.g., compiler/linter-only or Infer-based) to better isolate HapRepair’s contributions?
Please report cost/latency metrics for the LLM stages (tokens per prompt/patch, median time per patch and per project per iteration) and the frequency of rollbacks. This would substantiate the “cost-controllable” claim.
For HomeCheck precision: could you add 95% confidence intervals for the 0-FP estimate, and per-rule precision statistics? Are there rules you exclude due to known noisiness?
How are overlapping context groups resolved when repair actions conflict (e.g., different patches for merged defects)? Do you queue or batch patches, and how do you ensure atomicity/consistency across iterations?
Can you share more about the 2% regressions in the human study (root causes, rule categories, whether they passed the CFG guardrail), and whether additional guardrails might prevent them?
How sensitive are results to the knowledge base composition? Have you tried reducing examples per rule or mixing synthetic examples to test robustness to curation bias?
Beyond ArkTS, have you conducted preliminary trials in another language with a weaker/noisier oracle to test external validity of the four oracle properties?
Overall Assessment
This is a timely and well-executed systems paper addressing a practical and important problem: how to make automated code repair work in low-resource ecosystems without tests or large corpora. The central design principle—precision-first static oracles for where-to-fix and LLMs for how-to-fix—feels right and is instantiated thoughtfully through HomeCheck, context extraction/combination, and RAG. The empirical results on ArkTS/OpenHarmony are strong within the chosen metric (static-oracle compliance), and the ablation studies are informative. The paper is also careful to caveat what is and is not being claimed.

However, for a TOSEM-level contribution, I see two main gaps to address before publication: (1) stronger baselines and positioning against closely related static-oracle–guided APR pipelines (and template-only auto-fixes), and (2) deeper evaluation on cost and safety (token/time budgets, rollback rates, per-rule breakdowns, and additional semantic checks where feasible). Improving the precision characterization of HomeCheck and tightening presentation artifacts would also help.

Overall, I recommend a major revision. The work is promising and likely valuable to the community; with stronger baselines, richer cost/safety analysis, and clearer positioning, it could merit acceptance.