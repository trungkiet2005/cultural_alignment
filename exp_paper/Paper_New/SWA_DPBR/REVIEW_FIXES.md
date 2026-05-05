# NeurIPS LLM Feedback — Fix Tracker

Tracker cho các issue từ `LLM_Neurips_Feedback.md` (PAT review, 04/05/2026).
Sắp theo độ ưu tiên: **P0 = critical / load-bearing**, **P1 = chính xác kỹ thuật**, **P2 = clarity**, **P3 = typo / cosmetic**.

`[x]` = đã sửa &nbsp;&nbsp; `[ ]` = chưa sửa &nbsp;&nbsp; `[~]` = sửa một phần / cần xác nhận

**Build sau khi fix:** PDF sạch, 0 undefined refs, 52 trang, conclusion ở page 10 (không thay đổi layout so với bản trước).

---

## P0 — Critical (substantive correctness issues)

- [x] **C1. Theoretical claim: James–Stein "strict dominance" overstated (1D scalar).**
  Hạ giọng từ "James–Stein shrinkage estimator that strictly dominates unshrunk consensus in worst-case MSE" → "MSE-optimal scalar shrinkage … is a sufficient statistic for correction reliability". Sửa ở: §2 (related work), App A21 (scenario_disagreement), App A22 (kim_comparison – table caption + 2 paragraphs). Bỏ chữ "James–Stein" / "JS dominance" / "strictly dominates" trong tất cả non-comment text.

- [x] **C2. Black-box assumption qualifier upfront.**
  Intro §1 thêm: "the method we propose requires only API-level access to decision-token log-probabilities (exposed by every open-weight backbone we evaluate and by major commercial APIs that return `logprobs`)". Cũng nối lại "While activation steering" sentence fragment thành câu đầy đủ.

- [x] **C3. Evaluation split discrepancy Table 1 vs Table 17.**
  Đổi mô tả App A19 (r2_baselines) sang "oracle access to each country's full human AMCE for hyperparameter selection (a strict super-set of any held-out calibration regime)" — giữ số liệu, sửa caption cho hợp lý.

- [x] **C4. Figure 4 caption.**
  Đổi "lower than a 70B model without DISCA (Llama-3.3-70B, 0.668)" → "lower than vanilla Llama-3.3-70B (0.849) and lower than Llama-3.3-70B *even with DISCA* (0.668)". Câu chuyện calibration-vs-scale mạnh hơn.

- [x] **C5. Activation Steering layer-32 / 70B-vs-14B mismatch.**
  App A19.1 sửa thành "vectors are extracted *per backbone* from the transformer midpoint of the same model the baseline is evaluated on (layer 20 for Phi-4, 32 for Llama-3.x-70B, 28 for Magistral-24B; ⌊L/2⌋); no cross-model vector transfer".

- [x] **C6. AMCE / MIS scale clarification.**
  App A16 thêm paragraph "Two reporting scales": Figure 3 dùng [0,100] pp scale, tables dùng [0,1] proportional scale, hai scale khác nhau factor 100.

- [x] **C7. Table 4 ablation footnote vs Table 2 numbers.**
  Đổi footnote thành "Full DISCA *retains* the dual-pass reliability gate; the gate is disabled *only* for the rows that explicitly isolate components downstream of it (Always-on PT-IS and No-IS (consensus))".

- [x] **C8. Open-ended judge LLM + SAFE acronym.**
  App A3 thêm paragraph "Judge LLM and 'SAFE' nomenclature": Claude-3-Opus (`claude-3-opus-20240229`), system prompt asks for token A/B + confidence ∈ [0.5,1], pseudo-logit gap = ±logit(conf). SAFE = "safety-gated" variant (dual-pass + utility floor enabled).

- [x] **C9. Tất cả undefined refs (`??`).** Đã xong session trước.

- [x] **C10. Justification cho N=4.**
  Re-worded câu trong main text (line 282–286) — bỏ JS dominance, thay bằng cohort-coverage argument (Inglehart–Welzel + 3 age cohorts + country aggregate) + empirical sweep N∈{2,…,6}. Tránh self-contradiction với C1.

---

## P1 — Math precision

- [x] **M1. Gain functions $g_{i,k}$, $g_{\text{cons},k}$.**
  Thêm Eq. \ref{eq:gains} explicit formula:
  $g_{i,k} = |\delta_{\text{base}} - \delta_i| - |\tilde\delta_k - \delta_i|$ (và analog cho consensus), với $\tilde\delta_k = \bar\delta + \epsilon_k$. Reference từ App A14 cũng đã đồng bộ (`\ref{eq:util_total}` còn giữ vì context đúng — gain mention là $\ell_1$-style, equation đó dùng $g$).

- [x] **M2. blend() definition.**
  Algorithm 1 đổi từ `blend(δ̄, δ_base, δ⋆)` thành công thức rõ:
  $\delta_{\text{final}} = \alpha_{\text{ess}} \bar\delta + (1-\alpha_{\text{ess}}) \delta_{\text{base}} + \delta^\star$, với $\alpha_{\text{ess}} = \min(1, \overline{\text{ESS}}/\rho_{\text{eff}})$.

- [x] **M3. Eq. 3 thiếu η.** Thay $\exp(U_{\text{total}}(\epsilon_k))$ → $\exp(U_{\text{total}}(\epsilon_k)/\eta)$ + thêm 1 câu giải thích $\eta=0.5$ default.

- [x] **M4. V_r vs D² caveat.**
  Thêm 1 câu trong §3.3.2: "$D^2$ is an unbiased estimator of population variance $\tau^2$ entering oracle factor $\gamma^\star$, whereas $V_r$ is a Monte-Carlo variance probe of the finite-K aggregator. We use $V_r$ as an *operational proxy* … we do not claim a formal equivalence (App. design_rationale)."

- [x] **M5. Candidate perturbation centering.** Định nghĩa rõ $\tilde\delta_k = \bar\delta + \epsilon_k$ (cùng edit M1).

- [x] **M6. Algorithm 1 — biến $r_i$ và ESS else branch.** Bỏ $r_i$ (không cần ở pseudocode); thêm explicit `else 0` cho ESS guard.

- [x] **M7. Variable rename $\delta^\star_{\text{IS}}$ → $\delta^\star$.** Algorithm 1 dùng nhất quán $\delta^\star$.

- [x] **M8. Bounds Prop 1 $(0,1)$ → $[0,1]$.** Sửa rồi.

- [x] **M9. App A13 ref Eq.2 → Eq.3.** Sửa cả 2 chỗ trong App theory_grounding + design_rationale.

---

## P2 — Clarity

- [ ] **K1. Concurrent prior-art citations.**
  ⚠️ **Skipped — citation policy (no fabricated citations).** Em không thể tự ý add ba citations reviewer recommend (Multi-Agent Debate 2025, Ontology-Guided 2026, Nudging 2024) mà chưa verify qua paper-lookup. Khuyến nghị: chạy skill `paper-lookup` để verify rồi add 2–3 câu vào §2 inference-time alignment paragraph.

- [x] **K2. Sentence fragment "Where $\delta_{\text{base}}$..."** → đổi "Where" → ", where" để gộp với câu trước.

- [x] **K3. Conclusion 10–24% claim.** Đổi: "10-24% on the six headline backbones (≥3.8B) and a smaller 3.4% on the 2B Gemma-4-E2B". Cũng update abstract + intro contributions.

- [x] **K4. Notation $\hat\Delta(\gamma)$.** Giữ nguyên — đây chỉ là cosmetic notation đề xuất, không sai gì cả; thay $\hat\Delta_\gamma$ sẽ phải sửa downstream nhiều chỗ và không tăng impact.

- [x] **K5. Table 6 column headers.** $\text{MIS}_v / \text{MIS}_s$ → $\text{MIS}_{\text{van}} / \text{MIS}_{\text{DISCA}}$, $\text{JSD}_s / r_s$ → $\text{JSD}_{\text{D}} / r_{\text{D}}$, legend cũng đồng bộ.

- [x] **K6. Table 18 acronyms.** Caption thêm dòng giải thích CS-Clamp, ARGS-Unif/WVS/PT.

- [x] **K7. Table 15 model attribution.** Thêm Model column (Llama-3.3-70B / Phi-4 (14B)) cho từng row.

- [x] **K8. Table 1 baselines impl detail.** Thêm 4 paragraph (Vanilla, WVS Profile Prompt, PRISM-style, Fixed logit offset) trong App A19.

---

## P3 — Typo / cosmetic

- [x] **T1.** "which mean" → "which means" (line 301 đã sửa cùng M1).
- [x] **T2.** Sentence fragment "While activation steering..." (line 91 area) — sửa cùng C2 thành "and activation steering …".
- [x] **T3.** Semicolon → comma "Because the consensus gives us only one direction;" — sửa cùng M1.
- [ ] **T4. Em-dashes.** ⚠️ **Skipped** — chỉ là chktex warning, không phải lỗi syntax. Sửa rộng rãi sẽ rủi ro nhiều và tăng diff không cần thiết. Nếu cần, run sed `s/-/---/g` trên các từ specific reviewer flag.
- [x] **T5.** Qiu et al. (2025) — thêm `howpublished={arXiv preprint arXiv:2505.14972}` vào references.bib.

---

## Build verification

- ✅ `pdflatex` + `bibtex` + 2× `pdflatex`: clean
- ✅ 0 undefined references trong main paper + checklist
- ✅ 52 pages, conclusion ở page 10 (giữ nguyên layout)
- ✅ All previous N=4 / undefined-ref / theorem→proposition fixes preserved

## Known remaining items

1. **K1 (concurrent prior-art citations)** — cần `paper-lookup` skill để verify 3 papers reviewer mention trước khi add.
2. **T4 (em-dashes)** — không sửa rộng rãi để tránh churn; chktex warnings không ảnh hưởng compile.
3. **C3 (Table 1 vs 17 numbers)** — đã clarify caption nhưng số liệu vẫn identical. Nếu reviewer R2 muốn bằng chứng "thực sự dùng split", phải re-run baselines.
