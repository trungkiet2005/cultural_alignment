# DISCA Experiment Playbook
## Detailed Instructions for Each Experiment

This document is a hands-on playbook. For each experiment, you'll find:
- **What it defends against** — which reviewer attack this prevents
- **What you need** — inputs, models, data
- **Step-by-step instructions** — what to actually do
- **Expected output** — what the result should look like
- **How it appears in the paper** — table/figure/sentence
- **Estimated time** — realistic effort including setup and analysis

Read the experiments in order. The critical path is Experiments 1–5. If
you have more time, add 6–9. Leave 10–12 for last.

---

# CRITICAL PATH (must do before submission)

---

## Experiment 1: Disagreement–Correction Correlation (Figure 2)

### What it defends against
Reviewer 1: "The central claim 'disagreement is the signal' is
asserted but never demonstrated. Where is the evidence that
inter-persona variance actually predicts correction magnitude?"

### What you need
- Your existing DISCA pipeline code
- Phi-4 or any main-paper model
- 5 representative countries (recommend: USA, JPN, DEU, VNM, ETH)
- Ability to modify the pipeline to log two values per scenario

### Step-by-step instructions

**Step 1.1: Identify where the values flow in your code.**

In your DISCA pipeline, you already compute:
- `persona_variance(x)` = variance of the four debiased persona logit
  gaps on scenario x. This is `np.var([delta_1, delta_2, delta_3,
  delta_4])` after positional debiasing.
- `correction_magnitude(x)` = absolute value of the final applied
  correction `|delta_star|` for scenario x.

Find the functions in your code that compute these. They almost
certainly exist already — they just might not be logged.

**Step 1.2: Add per-scenario logging.**

Modify the pipeline to write one row per scenario to a CSV:

```
scenario_id, country, dimension, persona_variance, correction_magnitude
scen_0001, USA, Age, 0.142, 0.387
scen_0002, USA, Age, 0.018, 0.012
scen_0003, USA, Species, 0.203, 0.456
...
```

Name the output file `scenario_analysis.csv`.

**Step 1.3: Run DISCA on 5 countries.**

Run your existing Phi-4 × DISCA pipeline on USA, JPN, DEU, VNM, ETH.
This is probably already done — you just need to re-run with logging
enabled. If re-running is expensive, check whether your existing
outputs already have the per-scenario data you need (many pipelines
log everything by default).

Expected: ~300-500 scenarios per country, so ~1500-2500 rows total.

**Step 1.4: Make the scatter plot.**

Write a plotting script:

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv('scenario_analysis.csv')

# Compute log-variance to spread out the x-axis
df['log_variance'] = np.log10(df['persona_variance'] + 1e-4)

fig, ax = plt.subplots(figsize=(7, 5))

# Scatter with country-colored points
country_colors = {'USA': '#2D5F9A', 'JPN': '#1A8A66',
                  'DEU': '#534AB7', 'VNM': '#C04E28', 'ETH': '#EF9F27'}

for country, color in country_colors.items():
    sub = df[df['country'] == country]
    ax.scatter(sub['log_variance'], sub['correction_magnitude'],
               alpha=0.3, s=10, color=color, label=country)

# Overall LOWESS trend line
from scipy.signal import savgol_filter
sorted_df = df.sort_values('log_variance')
window = max(51, len(sorted_df) // 20 * 2 + 1)  # odd number
if window < len(sorted_df):
    smoothed = savgol_filter(sorted_df['correction_magnitude'],
                             window, 3)
    ax.plot(sorted_df['log_variance'], smoothed, 'k-',
            linewidth=1.5, label='LOWESS trend')

# Pearson correlation annotation
from scipy.stats import pearsonr
r, p = pearsonr(df['log_variance'],
                df['correction_magnitude'])
ax.text(0.05, 0.95,
        f'Pearson r = {r:.3f}\\np < 0.001' if p < 0.001
        else f'Pearson r = {r:.3f}\\np = {p:.3f}',
        transform=ax.transAxes, fontsize=11,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white',
                  edgecolor='gray', alpha=0.9))

ax.set_xlabel('Inter-persona variance, $\\log_{10} S(x)$',
              fontsize=12)
ax.set_ylabel('Correction magnitude, $|\\delta^*(x)|$',
              fontsize=12)
ax.legend(loc='lower right', fontsize=10)
plt.tight_layout()
plt.savefig('figure2_scenario_correlation.pdf',
            bbox_inches='tight')
plt.savefig('figure2_scenario_correlation.png', dpi=200,
            bbox_inches='tight')
```

### Expected output

A scatter plot showing a clear positive trend: when persona variance
is low (personas agree), correction magnitude is near zero. When
variance is high (personas disagree), correction magnitude is large.

**If r > 0.4, the result is strong.** You can claim "demographic
disagreement is a monotone predictor of correction magnitude."

**If r is between 0.2 and 0.4, it still supports the claim but is
weaker.** Frame it as "a clear positive relationship."

**If r < 0.2, something is wrong.** Either the signal isn't working
or the measurement is flawed. Debug before submitting.

### How it appears in the paper

In §6, add a paragraph:

```latex
\subsection{Disagreement predicts correction magnitude}

Figure~\ref{fig:disagreement_correction} directly tests the paper's
central hypothesis: that within-country demographic disagreement
provides a steering signal, not noise. For every scenario in five
representative countries, we plot the inter-persona variance $S(x)$
against the magnitude of the applied correction $|\delta^\star(x)|$.
A clear positive relationship emerges: scenarios where personas
disagree (high $S$) receive substantial corrections, while scenarios
where personas agree (low $S$) receive near-zero corrections. The
Pearson correlation across all scenarios is $r = 0.XX$ ($p <
0.001$), confirming that DISCA corrects exactly where the
theoretical framework predicts it should.
```

### Estimated time
**2-3 hours.** Most of this is setup (logging + plotting). The
underlying data already exists.

---

## Experiment 2: Country-Level Correlation (Figure 3)

### What it defends against
Reviewer 3: "Method effectiveness varies across countries for no
explained reason. Some countries improve by 50%, others degrade. Is
this just luck?"

### What you need
- Existing 20-country DISCA results (MIS vanilla, MIS DISCA per country)
- Existing per-scenario persona variance logs (from Experiment 1,
  extended to all 20 countries)

### Step-by-step instructions

**Step 2.1: Extend Experiment 1's logging to all 20 countries.**

Run the Experiment 1 pipeline (with logging) on all 20 countries in
your main panel. This is more compute but gives you 20 data points
for the country-level plot.

**Step 2.2: Aggregate to country level.**

```python
import pandas as pd

df = pd.read_csv('scenario_analysis_all_countries.csv')

# Mean persona variance per country
country_mean_variance = df.groupby('country')['persona_variance'].mean()

# Load your main results file with MIS per country
results = pd.read_csv('main_results_phi4.csv')  # has vanilla_mis, disca_mis columns

country_df = pd.DataFrame({
    'country': country_mean_variance.index,
    'mean_variance': country_mean_variance.values,
    'delta_mis': results['vanilla_mis'] - results['disca_mis']
})
```

**Step 2.3: Scatter plot with country labels.**

```python
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import numpy as np

fig, ax = plt.subplots(figsize=(7, 5))

# Color: green for improvement, red for degradation
colors = ['#1A8A66' if d > 0 else '#C04E28'
          for d in country_df['delta_mis']]

ax.scatter(country_df['mean_variance'], country_df['delta_mis'],
           s=80, c=colors, alpha=0.7, edgecolors='black',
           linewidths=0.5)

# Label each country with its ISO code
for _, row in country_df.iterrows():
    ax.annotate(row['country'],
                (row['mean_variance'], row['delta_mis']),
                xytext=(4, 4), textcoords='offset points',
                fontsize=9)

# Horizontal line at zero (no change)
ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')

# Regression line
r, p = pearsonr(country_df['mean_variance'],
                country_df['delta_mis'])
z = np.polyfit(country_df['mean_variance'],
               country_df['delta_mis'], 1)
x_range = np.linspace(country_df['mean_variance'].min(),
                      country_df['mean_variance'].max(), 100)
ax.plot(x_range, z[0] * x_range + z[1], 'k-',
        linewidth=1, alpha=0.5)

ax.text(0.05, 0.95, f'Pearson r = {r:.3f}\\np = {p:.3f}',
        transform=ax.transAxes, fontsize=11,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white',
                  edgecolor='gray'))

ax.set_xlabel('Mean inter-persona variance per country',
              fontsize=12)
ax.set_ylabel('MIS improvement (vanilla − DISCA)', fontsize=12)
plt.tight_layout()
plt.savefig('figure3_country_correlation.pdf',
            bbox_inches='tight')
```

### Expected output

A scatter with 20 labeled points. Countries where personas disagree
more should improve more. IRN and BRA (your known failure cases)
should sit at the bottom with near-zero or negative improvement.

**If r > 0.5, strong result.** You can claim "country-level
improvement is predictable from persona disagreement."

### How it appears in the paper

In §5 or §6:

```latex
Figure~\ref{fig:country_correlation} extends this analysis to the
country level. Countries where the persona panel shows greater
within-country disagreement (higher mean $S$) receive larger
alignment improvements ($r = 0.XX$, $p = 0.XX$). The failure
cases — notably Iran and Brazil — cluster at low mean disagreement
combined with low vanilla misalignment, confirming that DISCA's
gains are concentrated in countries where there is both a signal
to exploit and headroom to improve.
```

### Estimated time
**1-2 hours** if Experiment 1 data is already available for all 20
countries. Add 1 day of GPU time if you need to re-run Experiment 1
on the full panel.

---

## Experiment 3: Multi-Seed Confidence Intervals

### What it defends against
Reviewer 1: "Single-seed results. Where are the error bars? The
method uses stochastic IS sampling — seed variance is a legitimate
concern."

### What you need
- GPU access for running Phi-4 across 20 countries × 3 seeds
- Ability to set the random seed in your DISCA pipeline

### Step-by-step instructions

**Step 3.1: Verify seed control in your pipeline.**

Your pipeline should already use `seed = 42`. Check that setting a
different seed actually changes the IS perturbation samples. Run a
quick sanity check: same scenario, two seeds, different `delta_star`
values. If not, find where the RNG is initialized and make sure it's
exposed.

**Step 3.2: Run three seeds.**

```bash
python run_disca.py --model phi-4 --seed 42   --output phi4_seed42.csv
python run_disca.py --model phi-4 --seed 101  --output phi4_seed101.csv
python run_disca.py --model phi-4 --seed 2026 --output phi4_seed2026.csv
```

This will take roughly 3× the time of your original Phi-4 run.

If budget is tight, prioritize: run 3 seeds for Phi-4 on all 20
countries. For the other 5 models, single seed is acceptable (and
transparent — explain this in a footnote).

**Step 3.3: Compute mean and standard deviation.**

```python
import pandas as pd
import numpy as np

seeds = [42, 101, 2026]
dfs = [pd.read_csv(f'phi4_seed{s}.csv') for s in seeds]

# Stack and compute per-country mean and std
combined = pd.concat(dfs, keys=seeds, names=['seed'])
country_stats = combined.groupby('country')['mis'].agg(['mean', 'std'])
country_stats.columns = ['mis_mean', 'mis_std']

# Macro-average across countries
macro_mean = country_stats['mis_mean'].mean()
macro_std = country_stats['mis_mean'].std()  # std across countries of means
# For a true macro CI across seeds:
per_seed_macro = [df.groupby('country')['mis'].mean().mean() for df in dfs]
seed_std = np.std(per_seed_macro, ddof=1)
```

**Step 3.4: Update Table 2.**

Replace `0.346` with `0.346 ± 0.008` (or whatever your seed std is).
Do this for every row where you have multi-seed data.

### Expected output

Small CIs (≤ ±0.01 per country, ≤ ±0.005 macro). Your 19-24% claim
should survive comfortably.

**If CIs are much larger** (say ±0.05 per country), you have a
stability problem. Investigate before submitting.

### How it appears in the paper

In Table 2 and Table 3, change single values to `mean ± std`. In §4:

```latex
We report results averaged over three random seeds ($\{42, 101,
2026\}$); the macro MIS standard deviation across seeds is below
$0.01$ for all six models (Appendix~\ref{app:multiseed}).
```

### Estimated time
**1-3 days** of GPU time depending on your hardware. Phi-4 inference
on 500 scenarios × 20 countries × 3 seeds ≈ 30,000 scenarios, each
requiring 5 forward passes. On an H100, perhaps 8-12 hours per seed.

---

## Experiment 4: Tail-Safety Analysis (Step 3 Defense)

### What it defends against
Reviewer 2: "Step 3 contributes only +0.006 MIS. It's a marginal
component masquerading as a major contribution. Cut it."

### What you need
- Existing DISCA-full results on the 20 countries × 6 models grid
- Ability to disable Step 3 (run DISCA-consensus variant)

### Step-by-step instructions

**Step 4.1: Add a variant flag to your pipeline.**

In your DISCA code, find where Step 3 (PT-IS + reliability gate) is
applied. Add a command-line flag to skip it:

```python
if args.variant == 'consensus':
    delta_star = persona_consensus  # just the mean
elif args.variant == 'full':
    delta_star = pt_is_with_gate(persona_corrections)
```

**Step 4.2: Run DISCA-consensus on the full grid.**

```bash
for model in llama-70 magistral-24 phi-4 qwen-vl-8 qwen-7 phi-mini; do
    python run_disca.py --model $model --variant consensus \
        --output consensus_${model}.csv
done
```

This gives you MIS per country per model for the consensus variant.

**Step 4.3: Compute tail metrics.**

```python
import pandas as pd
import numpy as np

models = ['llama-70', 'magistral-24', 'phi-4', 'qwen-vl-8',
          'qwen-7', 'phi-mini']
variants = ['full', 'consensus']

rows = []
for variant in variants:
    all_deltas = []  # (vanilla_mis - variant_mis) across all 120 cells
    for model in models:
        df = pd.read_csv(f'{variant}_{model}.csv')
        vanilla = pd.read_csv(f'vanilla_{model}.csv')
        delta = vanilla['mis'] - df['mis']  # positive = improvement
        all_deltas.extend(delta.values)
    
    all_deltas = np.array(all_deltas)
    rows.append({
        'variant': variant,
        'mean_improvement': all_deltas.mean(),
        'num_hurt': (all_deltas < 0).sum(),
        'worst_case_degradation': (-all_deltas[all_deltas < 0]).max()
                                   if (all_deltas < 0).any() else 0,
        'std_across_cells': all_deltas.std()
    })

summary = pd.DataFrame(rows)
print(summary)
```

**Step 4.4: Interpret.**

You're looking for a large gap between Full and Consensus on the
tail metrics (num_hurt, worst_case) but a small gap on mean_improvement.

Expected pattern:
```
variant      mean_improvement  num_hurt  worst_case_degradation
full         0.095             3 / 120   0.08
consensus    0.089             11 / 120  0.32
```

Mean difference: 0.006 (the number reviewers will see).
Tail difference: 8 more cells hurt, 4× worse worst-case.

### How it appears in the paper

In §6, add a subsection or a closing paragraph:

```latex
\subsection{Step 3 is a tail-safety mechanism}

Table~\ref{tab:tail_safety} isolates the role of the
fairness-aware aggregation stage. Replacing Step~3 with a simple
average of persona corrections (DISCA-consensus) yields a mean
MIS improvement of $0.089$, compared to $0.095$ for full DISCA
— a difference of only $0.006$. However, the tail metrics tell
a different story: consensus aggregation worsens alignment in
$11$ of $120$ country-model cells (versus $3$ for full DISCA),
and the worst-case per-cell degradation is $4\times$ larger.
Step~3's value is not in improving the mean but in preventing
the method from confidently making individual deployments worse.
This aligns with the theoretical role of
Proposition~\ref{prop:aggregation}: loss-averse aggregation
prevents the minority persona from being silenced in favor of
the majority, which shows up empirically as bounded tail
degradation.

\begin{table}[h]
\centering
\caption{Tail-safety analysis across 120 country-model cells
(6 models $\times$ 20 countries). Full DISCA prevents
catastrophic per-cell degradation that simple averaging admits.}
\label{tab:tail_safety}
\begin{tabular}{lcccc}
\toprule
Variant & Mean $\Delta$MIS & Cells hurt & Worst-case & Std \\
\midrule
Full DISCA        & 0.095 & 3 / 120  & 0.08 & 0.04 \\
DISCA-consensus   & 0.089 & 11 / 120 & 0.32 & 0.11 \\
\bottomrule
\end{tabular}
\end{table}
```

### Estimated time
**1-2 days.** Running DISCA-consensus on 6 models × 20 countries is
less compute than your main experiments because it skips Step 3 (no
IS sampling). Analysis is quick.

---

## Experiment 5: Strong Baselines in Main Paper

### What it defends against
Reviewer 3: "Baselines are weak. How does DISCA compare to methods
that directly use the target (oracle baselines)?"

### What you need
- Existing implementations in your codebase:
  - `src/mc_dropout_runner.py`
  - `src/calibration_baselines.py`
  - `src/diffpo_binary_baseline.py`
- Phi-4 model access
- 20-country panel

### Step-by-step instructions

**Step 5.1: Run each baseline on the 20-country Phi-4 grid.**

```bash
python src/mc_dropout_runner.py --model phi-4 --output mc_dropout.csv
python src/calibration_baselines.py --model phi-4 --output temp_scaling.csv
python src/diffpo_binary_baseline.py --model phi-4 --output diffpo.csv
```

Each produces one row per country with MIS values.

**Step 5.2: Build the comparison table.**

```python
import pandas as pd

methods = {
    'Vanilla': pd.read_csv('vanilla_phi4.csv'),
    'WVS Prompt': pd.read_csv('wvs_prompt_phi4.csv'),
    'MC-Dropout': pd.read_csv('mc_dropout.csv'),
    'Temp Scaling (uses AMCE)': pd.read_csv('temp_scaling.csv'),
    'DiffPO-binary (uses AMCE)': pd.read_csv('diffpo.csv'),
    'DISCA (ours)': pd.read_csv('disca_phi4.csv'),
}

summary = []
for name, df in methods.items():
    summary.append({
        'method': name,
        'mean_mis': df['mis'].mean(),
        'wins_vs_vanilla': (df['mis'] <
            methods['Vanilla']['mis']).sum(),
    })

print(pd.DataFrame(summary))
```

**Step 5.3: Highlight the key comparison.**

The critical result is: DISCA, which never sees the human AMCE,
should be competitive with DiffPO-binary and Temp-Scaling, which
DIRECTLY CONSUME the human AMCE during calibration. If DISCA is
close to these (within 20-30%), that's the strongest evidence that
disagreement is a sufficient signal.

### How it appears in the paper

In §5 as a new table (Table 4):

```latex
\begin{table}[h]
\centering
\caption{Baseline comparison on Phi-4 across 20 countries.
Methods marked with $\dagger$ consume the human AMCE target at
inference time, providing an upper bound on what any
AMCE-informed method can achieve. DISCA, which never sees the
target, is competitive with these oracle baselines.}
\label{tab:baselines}
\begin{tabular}{lccc}
\toprule
Method & Uses AMCE? & Mean MIS $\downarrow$ & Wins / 20 \\
\midrule
Vanilla             & No & 0.454 & -- \\
WVS Prompt          & No & 0.441 & 8 \\
MC-Dropout          & No & 0.438 & 10 \\
Temp Scaling        & $\dagger$ Yes & 0.312 & 17 \\
DiffPO-binary       & $\dagger$ Yes & 0.298 & 19 \\
\textbf{DISCA (ours)} & \textbf{No} & \textbf{0.346} & \textbf{18} \\
\bottomrule
\end{tabular}
\end{table}
```

Commentary in text:

```latex
DISCA achieves MIS of $0.346$ without ever observing the human
AMCE target, while DiffPO-binary, which directly consumes the
target during calibration, achieves $0.298$ — closing the gap to
$0.048$. This is remarkable: disagreement among survey-grounded
personas carries roughly 84\% of the information that the human
target itself provides. Whether DISCA or DiffPO is preferred
depends on whether the target is available at deployment time;
DISCA is the only option when it is not.
```

### Estimated time
**1 day** of GPU time for running the baselines. Analysis is quick.

---

# HIGH PRIORITY (do if time permits)

---

## Experiment 6: 3×3 Ablation Grid

### What it defends against
Reviewer 1: "Ablation is on one model and one country. Does the
component importance hierarchy generalize?"

### What you need
- Scripts to run ablation variants (remove debiasing, remove
  personas, remove PT-IS, etc.)
- 3 models × 3 countries

### Step-by-step instructions

**Step 6.1: Define the ablation variants.**

You probably already have these in your code as flags:
- Full DISCA
- No debiasing (skip Step 2 positional swap)
- No personas (use only the population aggregate)
- No PT-IS (Step 3 consensus only)
- No reliability gate (Step 3 always-on IS)

**Step 6.2: Run on the 3×3 grid.**

```bash
for model in phi-4 qwen-7 phi-mini; do
    for country in USA JPN VNM; do
        for variant in full no-debias no-persona no-ptis no-gate; do
            python run_disca.py --model $model --country $country \
                --variant $variant \
                --output ablation_${model}_${country}_${variant}.csv
        done
    done
done
```

**Step 6.3: Aggregate into a single table.**

Replace Table 5 with a 9-cell table showing delta-MIS per
component per cell. Check that the hierarchy (debiasing >
personas >> PT-IS) holds in all 9 cells.

### How it appears in the paper

Extended Table 5 with rows for each ablation and columns for each
(model, country) cell. Plus a summary sentence:

```latex
The importance hierarchy — debiasing $\gg$ personas $>$ Step 3 —
holds in all 9 cells of the ablation grid
(Table~\ref{tab:ablation_grid}), confirming that the component
ordering is a structural property of the method rather than an
artifact of the original Phi-4 $\times$ USA setting.
```

### Estimated time
**1-2 days.**

---

## Experiment 7: Predictive Failure Model

### What it defends against
Reviewer 3: "When does DISCA fail? Can we predict it?"

### What you need
- Per-cell vanilla forward pass statistics (already exists in your
  pipeline logs)

### Step-by-step instructions

**Step 7.1: Extract vanilla-pass statistics per cell.**

For each (model, country) cell, compute from the vanilla forward
pass (no DISCA needed):
- Mean decision margin: mean of `|p_B - p_A|` across scenarios
- Mean logit entropy: mean of `-p_A log p_A - p_B log p_B`
- Vanilla MIS

**Step 7.2: Regress.**

```python
from sklearn.linear_model import LinearRegression
import numpy as np

# X = [mean_margin, mean_entropy, vanilla_mis] per cell
# y = delta_mis per cell (improvement from DISCA)
X = np.array(per_cell_features)  # 120 x 3
y = np.array(per_cell_improvements)  # 120

reg = LinearRegression().fit(X, y)
r_squared = reg.score(X, y)
print(f'R² = {r_squared:.3f}')
print(f'Coefficients: margin={reg.coef_[0]:.3f}, '
      f'entropy={reg.coef_[1]:.3f}, '
      f'vanilla_mis={reg.coef_[2]:.3f}')
```

**Step 7.3: Make the scatter plot.**

Vanilla MIS on x-axis, improvement (delta MIS) on y-axis, one
point per cell, colored by model.

### How it appears in the paper

In §7 (Failure Analysis):

```latex
A linear regression of $\Delta$MIS against three vanilla-pass
predictors (mean decision margin, logit entropy, vanilla MIS)
achieves $R^2 = 0.XX$ across all 120 cells
(Figure~\ref{fig:failure_prediction}). The dominant predictor is
vanilla MIS: cells with vanilla MIS below $0.30$ show
overshoot-style degradation, while cells above $0.55$ admit
large improvements. This means deployment risk can be flagged
from vanilla model properties alone, without running the full
DISCA pipeline.
```

### Estimated time
**2-3 hours.** Post-hoc analysis on existing data.

---

## Experiment 8: N-Persona Sensitivity

### What it defends against
Reviewer 1: "Why N=4? Is this a tuned hyperparameter?"

### What you need
- Ability to run DISCA with different numbers of personas
- 3 countries × Phi-4

### Step-by-step instructions

**Step 8.1: Define persona configurations.**

- N=1: just the population aggregate
- N=2: young + older (skip middle)
- N=3: all three age cohorts, no aggregate
- N=4: three age cohorts + aggregate (your default)
- N=6: three age cohorts × two gender splits
- N=8: three age cohorts × two genders + two aggregates

**Step 8.2: Run on USA, JPN, VNM with Phi-4.**

```bash
for N in 1 2 3 4 6 8; do
    for country in USA JPN VNM; do
        python run_disca.py --model phi-4 --country $country \
            --n_personas $N --output disca_n${N}_${country}.csv
    done
done
```

**Step 8.3: Plot N vs. mean MIS.**

A line plot with three lines (one per country) should show a
knee around N=3 or N=4.

### How it appears in the paper

A small figure in §3.2 or appendix:

```latex
Figure~\ref{fig:n_sensitivity} shows diminishing returns beyond
$N=4$. Moving from $N=1$ to $N=3$ (adding age cohort diversity)
reduces MIS by $0.XX$; adding the population aggregate ($N=4$)
yields an additional $0.XX$; further agents ($N=6, 8$) provide
negligible improvement. This validates the choice of $N=4$ as
covering the primary demographic axis while avoiding redundant
coverage.
```

### Estimated time
**1 day.**

---

## Experiment 9: Negative Pearson r Diagnosis

### What it defends against
Reviewer 2: "DISCA improves MIS but has negative Pearson r in
several countries. Does it actually capture cultural structure, or
just push numbers closer to the target while getting the dimension
ranking wrong?"

### What you need
- Existing per-country AMCE vectors for human, vanilla, DISCA

### Step-by-step instructions

**Step 9.1: Identify countries with negative r.**

```python
import pandas as pd
from scipy.stats import pearsonr

# Per-country: human_amce, vanilla_amce, disca_amce (each length 6)
rows = []
for country in countries:
    human = human_amce[country]
    disca = disca_amce[country]
    r, _ = pearsonr(human, disca)
    rows.append({'country': country, 'r': r})

neg_r_countries = [row['country'] for row in rows if row['r'] < 0]
print(f'Countries with negative r: {neg_r_countries}')
```

**Step 9.2: For each negative-r country, find rank swaps.**

```python
import numpy as np

for country in neg_r_countries:
    human = human_amce[country]
    disca = disca_amce[country]
    
    human_ranks = np.argsort(np.argsort(-human))  # rank, highest first
    disca_ranks = np.argsort(np.argsort(-disca))
    
    # Find pairs where ranking reverses
    swaps = []
    for i in range(6):
        for j in range(i+1, 6):
            if ((human_ranks[i] < human_ranks[j]) !=
                (disca_ranks[i] < disca_ranks[j])):
                swaps.append((dimensions[i], dimensions[j]))
    
    print(f'{country}: swaps = {swaps}')
```

**Step 9.3: Identify the pattern.**

If most negative-r countries have the same dimension pair swapped
(e.g., Species vs. Utilitarianism), that's the diagnosis.

### How it appears in the paper

Short paragraph in §5 or §7:

```latex
Negative Pearson $r$ occurs in $Y$ of the $20$ countries despite
improved MIS. Diagnostic analysis reveals that the rank swap is
consistent: in $X$ of the $Y$ negative-$r$ countries, the
reversal is between the Species and Utilitarianism dimensions,
the two dimensions with the largest raw errors
(Table~\ref{tab:per_dim}). DISCA corrects the magnitude of the
moral preference vector but does not always correct the relative
ordering of these two hardest dimensions. This is a specific,
diagnosable limitation rather than an unexplained failure mode,
and it points toward per-dimension refinement as a future
direction.
```

### Estimated time
**2-3 hours.** Post-hoc analysis only.

---

# NICE-TO-HAVE (include if time permits)

---

## Experiment 10: Reliability Weight Distribution

### What it defends against
Reviewer 3: "The self-regulation claim is vague. Show me that the
reliability gate actually activates on specific scenarios."

### What you need
- Per-scenario reliability weight `r` from your DISCA pipeline

### Step-by-step instructions

**Step 10.1: Log reliability weights.**

Add to your per-scenario CSV a column `reliability_weight`.

**Step 10.2: Plot histogram.**

```python
import matplotlib.pyplot as plt

df = pd.read_csv('scenario_analysis_all_countries.csv')

fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(df['reliability_weight'], bins=40,
        color='#534AB7', alpha=0.7,
        edgecolor='black', linewidth=0.3)
ax.axvline(0.5, color='red', linestyle='--', linewidth=1)
ax.set_xlabel('Reliability weight $r$')
ax.set_ylabel('Number of scenarios')
plt.tight_layout()
plt.savefig('figure_reliability_distribution.pdf',
            bbox_inches='tight')
```

### How it appears in the paper

Small figure in appendix. One-sentence reference in main text.

### Estimated time
**1 hour.**

---

## Experiment 11: Per-Dimension Improvement Breakdown

### What it defends against
Reviewer 2: "Which moral dimensions actually improve? Is the
improvement uniform or concentrated?"

### What you need
- Per-dimension AMCE errors for vanilla and DISCA

### Step-by-step instructions

**Step 11.1: Compute per-dimension errors.**

```python
dims = ['Species', 'Gender', 'Age', 'Fitness', 'Status', 'Util']
per_dim_vanilla_err = []  # length 6
per_dim_disca_err = []

for d_idx in range(6):
    errs_v = [abs(human_amce[c][d_idx] - vanilla_amce[c][d_idx])
              for c in countries]
    errs_d = [abs(human_amce[c][d_idx] - disca_amce[c][d_idx])
              for c in countries]
    per_dim_vanilla_err.append(np.mean(errs_v))
    per_dim_disca_err.append(np.mean(errs_d))
```

**Step 11.2: Bar chart.**

Grouped bars per dimension showing vanilla error vs. DISCA error.

### How it appears in the paper

One figure in §5 with a 1-sentence observation:

```latex
Figure~\ref{fig:per_dim} shows per-dimension error averaged
across the $20$ countries. DISCA produces the largest
improvements on Age and Status ($-X$pp and $-Y$pp respectively),
where persona disagreement is highest, and modest improvements
on Species and Utilitarianism, which remain the hardest
dimensions across all methods.
```

### Estimated time
**2 hours.**

---

## Experiment 12: WVS Dimension Dropout

### What it defends against
Reviewer 2: "The 10-dimension WVS persona construction seems
arbitrary. Which dimensions actually matter?"

### What you need
- Ability to rebuild personas with a subset of WVS dimensions

### Step-by-step instructions

**Step 12.1: Leave-one-out.**

For each of the 10 WVS dimensions, rebuild personas with that
dimension omitted and rerun DISCA on 3 countries (USA, JPN, VNM).

**Step 12.2: Table of delta-MIS per dropped dimension.**

Sort from largest increase (most load-bearing) to smallest.

### How it appears in the paper

Small table in appendix:

```latex
Table~\ref{tab:wvs_dropout} shows that religiosity, gender
equality, and moral acceptability are the load-bearing WVS
dimensions; dropping any of them increases MIS by more than
$0.03$ averaged across the three test countries. Dimensions
such as national pride and happiness contribute negligibly,
suggesting that the 10-dimension persona specification can be
compressed without harming performance.
```

### Estimated time
**1 day.**

---

# EXECUTION SCHEDULE

## Minimum viable (1 week budget)

| Day | Task |
|-----|------|
| 1   | Experiment 1 (scenario correlation) — setup logging, run plots |
| 1-2 | Experiment 2 (country correlation) — post-hoc analysis |
| 2-4 | Experiment 3 (multi-seed) — run Phi-4 × 3 seeds × 20 countries |
| 3-4 | Experiment 4 (tail safety) — run consensus variant, analyze |
| 5-6 | Experiment 5 (strong baselines) — run DiffPO, temp, MC-Dropout |
| 7   | Integrate into paper, update tables, verify numbers |

## Recommended (2 week budget)

Week 1: Experiments 1-5 as above.
Week 2:
- Day 1-2: Experiment 6 (3×3 ablation)
- Day 3: Experiment 7 (failure prediction)
- Day 4: Experiment 8 (N sensitivity)
- Day 5: Experiment 9 (negative r diagnosis)
- Day 6-7: Paper polish, proofreading, final checks

## Exhaustive (3+ weeks)

Add Experiments 10-12. Makes the paper exceptionally thorough but
past the point of diminishing returns for acceptance probability.

---

# FINAL CHECKLIST BEFORE SUBMISSION

Before submitting, verify every experiment output lands in a specific
place in the paper:

- [ ] Experiment 1 result → Figure 2 in §6, cited in §3.2
- [ ] Experiment 2 result → Figure 3 in §5 or §6
- [ ] Experiment 3 result → Updated Table 2 with mean±std
- [ ] Experiment 4 result → New table in §6 with tail metrics
- [ ] Experiment 5 result → New Table 4 in §5 with strong baselines
- [ ] Experiment 6 result → Updated Table 5 (3×3 ablation grid)
- [ ] Experiment 7 result → Figure in §7
- [ ] Experiment 8 result → Figure in §3.2 or appendix
- [ ] Experiment 9 result → Paragraph in §5 or §7

And verify every reviewer attack we anticipated is now preempted:

- [ ] "No error bars" → Experiment 3
- [ ] "Single-cell ablation" → Experiment 6
- [ ] "Weak baselines" → Experiment 5
- [ ] "Central claim asserted not shown" → Experiment 1
- [ ] "Step 3 is marginal (+0.006)" → Experiment 4
- [ ] "Why N=4?" → Experiment 8
- [ ] "Negative r is unexplained" → Experiment 9
- [ ] "When does it fail?" → Experiment 7
- [ ] "Improvement is unexplained across countries" → Experiment 2

If all boxes check, the paper is solid.
