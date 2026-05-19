# Ultimate Causal Analysis Report

**Project:** DSC245 Sector ETF Causal Inference Project  
**Data:** Sector ETF and S&P 500 log returns  
**Methods Used:** Conditional VAR Granger, PCMCI, XGBoost lagged-feature importance  
**Primary Question:** Which sectors most consistently lead or predict the S&P 500 (`^GSPC`) across daily, weekly, and monthly resolutions?

---

## 1. Executive Summary

This report uses three directed evidence streams:

- **Conditional VAR Granger (CVAR):** tests whether lagged `X` predicts `Y` after controlling for the other observed sectors and the market index.
- **PCMCI:** tests conditional lagged dependencies using Tigramite with ParCorr.
- **XGBoost:** identifies whether `X` is one of the top lagged predictors for `Y`.

Bivariate-only Granger testing is no longer used in the analysis, scoring, rankings, or interpretation.

The strongest sector leader for the S&P 500 is **Consumer Discretionary (`XLY`)**. The daily edge `XLY -> ^GSPC` is supported by all three remaining methods: CVAR, PCMCI, and XGBoost. The next strongest market-leading sector is **Technology (`XLK`)**, supported by both conditional statistical methods at the daily frequency.

The safest interpretation is:

> Sector lead-lag effects are strongest at the daily frequency. `XLY` is the clearest sector-level leading indicator for the S&P 500, followed by `XLK`. The evidence supports predictive temporal precedence, not definitive structural causation.

---

## 2. Research Questions

### 2.1 Do sectors predict the market, or does the market predict sectors?

The evidence goes in **both directions**, but with different interpretation.

For sector-to-market prediction, the strongest edges into `^GSPC` are:

| Resolution | Edge | Total Conditional Score | Core Causal Score | Evidence |
|---|---|---:|---:|---|
| Daily | `XLY -> ^GSPC` | 3 | 2 | CVAR, PCMCI, XGBoost |
| Daily | `XLK -> ^GSPC` | 2 | 2 | CVAR, PCMCI |
| Weekly | `XLV -> ^GSPC` | 2 | 1 | PCMCI, XGBoost |

For market-to-sector prediction, the strongest `^GSPC` source edges are:

| Resolution | Edge | Total Conditional Score | Core Causal Score | Evidence |
|---|---|---:|---:|---|
| Daily | `^GSPC -> XLE` | 2 | 2 | CVAR, PCMCI |
| Daily | `^GSPC -> XLB` | 2 | 2 | CVAR, PCMCI |
| Daily | `^GSPC -> XLK` | 2 | 1 | PCMCI, XGBoost |
| Daily | `^GSPC -> XLP` | 2 | 1 | PCMCI, XGBoost |
| Weekly | `^GSPC -> XLK` | 2 | 1 | PCMCI, XGBoost |
| Daily | `^GSPC -> XLF` | 2 | 1 | PCMCI, XGBoost |

Overall, the market also predicts sectors, especially at the daily frequency. This is economically plausible because broad market moves can transmit quickly into sector ETFs. However, the strongest individual sector-to-market edge is still `XLY -> ^GSPC`.

### 2.2 Is there a first mover sector that systematically leads all others?

Yes, the best candidate is **Consumer Discretionary (`XLY`)**.

Across all directed edges, excluding `^GSPC` as a source, the leading source sectors are:

| Rank | Source Sector | Total Conditional Score Sum | Core Causal Score Sum | CVAR Support | PCMCI Support | XGBoost Support |
|---:|---|---:|---:|---:|---:|---:|
| 1 | `XLY` | 30 | 15 | 8 | 7 | 15 |
| 2 | `XLE` | 22 | 6 | 1 | 5 | 16 |
| 3 | `XLV` | 19 | 9 | 1 | 8 | 10 |
| 4 | `XLB` | 14 | 7 | 1 | 6 | 7 |
| 5 | `XLI` | 13 | 9 | 3 | 6 | 4 |

`XLY` has the highest total support and the highest core support among sector source nodes. That makes it the strongest first-mover candidate in this dataset.

### 2.3 Which sectors causally lead the S&P 500 index?

Using the three-method conditional scoring rule, the sector ranking for leading `^GSPC` is:

| Rank | Sector | Total Conditional Score | Core Causal Score | CVAR Support | PCMCI Support | XGBoost Support |
|---:|---|---:|---:|---:|---:|---:|
| 1 | `XLY` | 4 | 2 | 1 | 1 | 2 |
| 2 | `XLK` | 3 | 2 | 1 | 1 | 1 |
| 3 | `XLV` | 3 | 1 | 0 | 1 | 2 |
| 4 | `XLB` | 2 | 1 | 0 | 1 | 1 |
| 5 | `XLE` | 2 | 0 | 0 | 0 | 2 |
| 6 | `XLI` | 1 | 1 | 0 | 1 | 0 |
| 7 | `XLP` | 1 | 0 | 0 | 0 | 1 |
| 8 | `XLF` | 0 | 0 | 0 | 0 | 0 |
| 9 | `XLU` | 0 | 0 | 0 | 0 | 0 |

The strongest causal-leading candidates for the S&P 500 are therefore:

1. **Consumer Discretionary (`XLY`)**
2. **Technology (`XLK`)**
3. **Healthcare (`XLV`)**
4. **Materials (`XLB`)**
5. **Energy (`XLE`)**

Because `XLE` has no core causal support for `^GSPC`, it should be treated as a predictive XGBoost signal rather than a strong causal candidate.

---

## 3. Data and Setup

The dataset is `sector_data_all.csv` and contains:

`XLF, XLB, XLE, XLI, XLK, XLV, XLY, XLP, XLU, ^GSPC`

Only log-return rows were used.

| Resolution | Observations | Variables |
|---|---:|---:|
| Daily | 6,882 | 10 |
| Weekly | 1,429 | 10 |
| Monthly | 330 | 10 |

ADF tests found that all return series were stationary at the 5% level. VAR lag selection used BIC, and the selected lag was 1 for daily, weekly, and monthly data.

---

## 4. Scoring and Ranking Rule

For each directed relationship `X -> Y`, the analysis computes:

`Total Conditional Score = CVAR + PCMCI + XGB`

where:

- `CVAR = 1` if conditional VAR Granger is significant after FDR adjustment, otherwise `0`
- `PCMCI = 1` if PCMCI is significant, otherwise `0`
- `XGB = 1` if XGBoost selects `X` as a top predictor for `Y`, otherwise `0`

It also computes:

`Core Causal Score = CVAR + PCMCI`

The ranking rule is:

1. Rank by `Total Conditional Score` descending.
2. Break ties by `Core Causal Score` descending.
3. Break remaining ties by lower conditional VAR adjusted p-value.
4. Break remaining ties by lower PCMCI p-value.

This rule is implemented in `phase4_ml_causal_analysis.py`.

---

## 5. Method Results

The three methods produced the following supported non-self edges:

| Resolution | Conditional VAR | XGBoost Top Edges | PCMCI |
|---|---:|---:|---:|
| Daily | 21 | 30 | 29 |
| Weekly | 2 | 30 | 15 |
| Monthly | 0 | 30 | 3 |

XGBoost always contributes 30 top edges per resolution because it selects the top 3 predictors for each of 10 targets. These are predictive rankings, not statistical significance tests.

The total conditional score distribution is:

| Resolution | Score 0 | Score 1 | Score 2 | Score 3 |
|---|---:|---:|---:|---:|
| Daily | 41 | 26 | 15 | 8 |
| Weekly | 53 | 27 | 10 | 0 |
| Monthly | 58 | 31 | 1 | 0 |

Daily data has the richest lead-lag structure. Weekly evidence is weaker but still meaningful. Monthly evidence is sparse and should be interpreted cautiously.

---

## 6. Strongest Directed Edges

The strongest edges are those with `Total Conditional Score = 3`, meaning all three methods agree:

| Resolution | Edge | CVAR Adj. P-Value | PCMCI P-Value |
|---|---|---:|---:|
| Daily | `XLY -> XLE` | 0.000002 | 0.000000027 |
| Daily | `XLY -> ^GSPC` | 0.000012 | 0.000000022 |
| Daily | `XLY -> XLK` | 0.001800 | 0.000003838 |
| Daily | `XLI -> XLE` | 0.003097 | 0.000005576 |
| Daily | `XLY -> XLP` | 0.003097 | 0.000022820 |
| Daily | `XLP -> XLU` | 0.018238 | 0.023659 |
| Daily | `XLY -> XLU` | 0.024395 | 0.011386 |
| Daily | `XLE -> XLV` | 0.041852 | 0.000015113 |

These are the most robust directed relationships in the project because they survive both conditional statistical methods and also appear in the XGBoost top-predictor set.

---

## 7. S&P 500 Leadership

The best-supported S&P 500 edge is:

> **Daily `XLY -> ^GSPC`**

Evidence:

- Conditional VAR adjusted p-value: `0.000012`
- PCMCI p-value: `2.20e-08`
- XGBoost selected `XLY` as a top predictor for `^GSPC`
- Total Conditional Score: `3`
- Core Causal Score: `2`

Interpretation:

Consumer Discretionary returns contain strong short-term predictive information about next-period S&P 500 returns. This makes economic sense because Consumer Discretionary is cyclical and sensitive to expectations about household spending, growth, credit conditions, and investor risk appetite.

The second strongest S&P 500 edge is:

> **Daily `XLK -> ^GSPC`**

Evidence:

- Conditional VAR adjusted p-value: `0.003097`
- PCMCI p-value: `0.001493`
- Total Conditional Score: `2`
- Core Causal Score: `2`

Technology is a major S&P 500 component, and the fact that `XLK -> ^GSPC` is supported by both conditional statistical methods makes it a strong market-leading candidate even without daily XGBoost support.

---

## 8. Interpretation by Time Resolution

### 8.1 Daily

Daily data produced the strongest evidence:

- 21 conditional VAR edges
- 29 PCMCI edges
- 30 XGBoost top edges
- 23 consensus edges with score at least 2
- 8 edges supported by all three methods

Most meaningful lead-lag structure appears at the daily frequency.

### 8.2 Weekly

Weekly data produced:

- 2 conditional VAR edges
- 15 PCMCI edges
- 30 XGBoost top edges
- 10 consensus edges with score at least 2

The strongest weekly sector-to-market edge is `XLV -> ^GSPC`, supported by PCMCI and XGBoost.

### 8.3 Monthly

Monthly data produced:

- 0 conditional VAR edges
- 3 PCMCI edges
- 30 XGBoost top edges
- 1 consensus edge with score at least 2

Monthly evidence is weak because the sample is much smaller and monthly aggregation removes much of the short-run lead-lag structure.

---

## 9. Limitations

These results should be interpreted carefully.

Conditional VAR Granger and PCMCI identify predictive temporal precedence under modeling assumptions. They do not prove structural causation.

XGBoost feature importance is predictive, not causal. It is useful as supporting evidence, especially when it agrees with CVAR or PCMCI.

Hidden macroeconomic confounders such as interest rates, inflation news, volatility, credit conditions, and monetary policy are not included in the dataset.

The selected lag is 1 for all frequencies. Some relationships may operate over longer horizons.

Monthly results are underpowered relative to daily and weekly results.

---

## 10. Final Answer

The sectors that most consistently lead or predict the S&P 500 are:

1. **Consumer Discretionary (`XLY`)**
2. **Technology (`XLK`)**
3. **Healthcare (`XLV`)**
4. **Materials (`XLB`)**
5. **Energy (`XLE`)**

The strongest single causal-leading edge is:

> **Daily `XLY -> ^GSPC`**

The strongest first-mover sector across the full network is:

> **Consumer Discretionary (`XLY`)**

The market also predicts sectors, especially at the daily frequency, but the strongest individual sector-to-market result remains `XLY -> ^GSPC`.

---

## 11. Files to Use

Read this report first:

- `Ultimate_Causal_Analysis_Report.md`

Main tables supporting the report:

- `phase3_outputs/tables/adf_stationarity_results.csv`
- `phase3_outputs/tables/var_lag_selection_results.csv`
- `phase3_outputs/tables/conditional_var_granger_fdr_adjusted_results.csv`
- `phase4_outputs/tables/consensus_edges.csv`
- `phase4_outputs/tables/strongest_consensus_edges.csv`
- `phase4_outputs/tables/sp500_overall_sector_leader_ranking.csv`
- `phase4_outputs/tables/xgboost_top_edges.csv`
- `phase4_outputs/tables/xgboost_model_performance.csv`
- `phase4_outputs/tables/pcmci_significant_edges.csv`
