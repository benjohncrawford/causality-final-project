# Sector ETF Lead-Lag Relationships and S&P 500 Predictability

**Course Project:** DSC245 Causal Inference Project  
**Dataset:** Sector ETF and S&P 500 log returns  
**Methods:** Conditional VAR Granger causality, PCMCI, and XGBoost lagged-feature importance  
**Main Research Question:** Which sector ETFs most consistently lead or predict the S&P 500 across daily, weekly, and monthly time resolutions?

---

## Abstract

This project studies whether sector ETF returns contain useful leading information about future S&P 500 returns. The analysis uses daily, weekly, and monthly log returns for nine sector ETFs and the S&P 500 index. To avoid relying on a single modeling approach, the project combines three forms of directed evidence: conditional VAR Granger causality, PCMCI conditional independence testing, and XGBoost lagged-feature importance. The strongest result is that Consumer Discretionary (`XLY`) consistently appears as a leading sector, especially at the daily frequency. The daily relationship `XLY -> ^GSPC` is supported by all three methods, making it the most robust sector-to-market finding in the project. Technology (`XLK`) is the second strongest market-leading sector, with support from both conditional statistical methods. Overall, the results suggest that sector-level lead-lag effects are most visible in short-horizon daily data, while weekly and monthly relationships are weaker. The findings should be interpreted as evidence of predictive temporal precedence, not proof of structural economic causation.

---

## 1. Introduction

Financial markets are often described as highly efficient, but short-run lead-lag relationships can still appear across related assets. Sector ETFs are especially interesting because they represent different parts of the economy. Some sectors may react more quickly to changing expectations about growth, inflation, consumer demand, interest rates, or risk appetite. If one sector moves before the broader market, its returns may contain information about future S&P 500 returns.

The purpose of this project is to identify which sectors, if any, act as leading indicators for the S&P 500. The project focuses on three related questions:

1. Do sectors predict the market, or does the market predict sectors?
2. Is there a first-mover sector that tends to lead the broader network?
3. Which sectors most strongly lead the S&P 500?

The analysis does not treat causality as a simple yes-or-no claim. In financial time series, many apparent causal relationships may be driven by shared macroeconomic shocks, investor behavior, or market microstructure. For that reason, this report uses the term "causal" in the limited Granger-style sense: a variable is considered causal if its past values improve prediction of another variable after conditioning on the rest of the observed system.

---

## 2. Data

The dataset is `sector_data_all.csv`. It contains log returns for the following variables:

`XLF, XLB, XLE, XLI, XLK, XLV, XLY, XLP, XLU, ^GSPC`

These represent Financials, Materials, Energy, Industrials, Technology, Healthcare, Consumer Discretionary, Consumer Staples, Utilities, and the S&P 500 index.

| Resolution | Observations | Variables |
|---|---:|---:|
| Daily | 6,882 | 10 |
| Weekly | 1,429 | 10 |
| Monthly | 330 | 10 |

Before running the causal analysis, Augmented Dickey-Fuller tests were used to check stationarity. All return series were stationary at the 5% significance level. This is important because VAR-based causality tests assume that the time series are stationary. VAR lag selection was based on BIC, and the selected lag was 1 for daily, weekly, and monthly data.

The lag choice also affects the interpretation. A lag of 1 means the analysis asks whether one period's sector return helps predict the next period's return of another asset. Therefore, the daily results capture next-day lead-lag behavior, while the weekly and monthly results capture one-week-ahead and one-month-ahead relationships.

---

## 3. Methodology

This project uses three methods because each one captures a different type of evidence.

### 3.1 Conditional VAR Granger Causality

Conditional VAR Granger causality tests whether lagged values of one variable improve prediction of another variable after controlling for all other variables in the system. This is stronger than bivariate Granger testing because it reduces the chance that a relationship is only caused by a third observed variable.

For example, if `XLY -> ^GSPC` is significant in the conditional VAR model, the interpretation is that past Consumer Discretionary returns contain incremental predictive information about future S&P 500 returns after accounting for the other sectors and the market index.

### 3.2 PCMCI

PCMCI is a causal discovery method designed for time series. In this project, PCMCI uses conditional independence tests to identify lagged relationships that remain significant after conditioning on other relevant variables. PCMCI is useful because it provides a second conditional statistical approach that is not identical to the VAR Granger framework.

When both conditional VAR and PCMCI support the same edge, that relationship is more convincing than an edge found by only one method.

### 3.3 XGBoost Lagged-Feature Importance

XGBoost is used as a machine learning prediction model. For each target variable, the model identifies the most important lagged predictors. Unlike the two statistical methods, XGBoost feature importance is not a causal test. It is included because it can capture nonlinear predictive patterns and can show whether a variable is useful for forecasting even if the statistical methods are more conservative.

Because XGBoost always selects top predictors, its evidence is interpreted as predictive support rather than causal support.

### 3.4 Scoring Rule

For each directed edge `X -> Y`, the project computes:

`Total Conditional Score = CVAR + PCMCI + XGB`

where each method contributes either 1 or 0. The project also computes:

`Core Causal Score = CVAR + PCMCI`

The core causal score is more conservative because it excludes XGBoost and only counts the two conditional statistical methods.

Edges are ranked using the following rule:

1. Higher total conditional score.
2. Higher core causal score.
3. Lower conditional VAR adjusted p-value.
4. Lower PCMCI p-value.

This scoring design gives the highest priority to relationships that are supported by multiple methods, especially relationships supported by both conditional statistical tests.

---

## 4. Results

### 4.1 Overall Method Results

The three methods produced different amounts of evidence at each time resolution.

| Resolution | Conditional VAR Edges | PCMCI Edges | XGBoost Top Edges |
|---|---:|---:|---:|
| Daily | 21 | 29 | 30 |
| Weekly | 2 | 15 | 30 |
| Monthly | 0 | 3 | 30 |

The daily data clearly contains the richest directed structure. This is not surprising because daily data has the largest sample size and because financial lead-lag effects are often short-lived. Weekly results still show some structure, but the evidence is weaker. Monthly results are the least reliable for causal interpretation because the sample size is much smaller and aggregation may remove short-term predictive signals.

### 4.2 Strongest Consensus Edges

The strongest edges are those with a total conditional score of 3, meaning conditional VAR, PCMCI, and XGBoost all support the same direction.

| Resolution | Edge | Conditional VAR Adj. P-Value | PCMCI P-Value |
|---|---|---:|---:|
| Daily | `XLY -> XLE` | 0.000002 | 0.000000027 |
| Daily | `XLY -> ^GSPC` | 0.000012 | 0.000000022 |
| Daily | `XLY -> XLK` | 0.001800 | 0.000003838 |
| Daily | `XLI -> XLE` | 0.003097 | 0.000005576 |
| Daily | `XLY -> XLP` | 0.003097 | 0.000022820 |
| Daily | `XLP -> XLU` | 0.018238 | 0.023659 |
| Daily | `XLY -> XLU` | 0.024395 | 0.011386 |
| Daily | `XLE -> XLV` | 0.041852 | 0.000015113 |

The most important pattern in this table is the repeated appearance of `XLY` as a source variable. Consumer Discretionary leads the S&P 500, Technology, Energy, Consumer Staples, and Utilities in the strongest consensus results. This suggests that Consumer Discretionary is not only related to the market index but also occupies an important position in the broader sector network.

Economically, this result is plausible. Consumer Discretionary companies are sensitive to consumer confidence, household spending expectations, borrowing conditions, and general optimism about future growth. When investors revise expectations about the economy, discretionary stocks may respond quickly because their earnings are closely tied to the business cycle.

### 4.3 Sector Leadership of the S&P 500

The overall sector ranking for predicting or leading the S&P 500 is:

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

The top result is Consumer Discretionary (`XLY`). The edge `XLY -> ^GSPC` appears at the daily frequency and is supported by all three methods:

| Edge | Total Conditional Score | Core Causal Score | Conditional VAR Adj. P-Value | PCMCI P-Value |
|---|---:|---:|---:|---:|
| `XLY -> ^GSPC` | 3 | 2 | 0.000012 | 0.000000022 |

This is the strongest single sector-to-market finding in the project. The interpretation is that yesterday's Consumer Discretionary return helps predict today's S&P 500 return, even after controlling for the other sector ETFs and the market system. Since both conditional statistical methods support the edge, the result is stronger than a purely machine learning feature-importance result.

Technology (`XLK`) is the second strongest sector leader for the S&P 500. The edge `XLK -> ^GSPC` is supported by conditional VAR and PCMCI at the daily frequency. This also makes economic sense because Technology has a large weight in the S&P 500 and often reflects investor expectations about growth, innovation, and risk appetite. However, Technology does not dominate the full network as strongly as Consumer Discretionary.

Healthcare (`XLV`) ranks third. Its support is weaker because it has only one core causal signal, but it appears in PCMCI and XGBoost evidence. Healthcare may provide useful predictive information because it has both defensive and growth-related characteristics, but the evidence is less consistent than for `XLY` and `XLK`.

Energy (`XLE`) is a useful example of why the core causal score matters. It receives predictive support from XGBoost, but it has no core causal support for leading the S&P 500. Therefore, it should be interpreted as a forecasting signal rather than a strong causal-leading candidate.

### 4.4 Sector-to-Market Versus Market-to-Sector Direction

The analysis finds evidence in both directions. Some sectors predict the S&P 500, but the S&P 500 also predicts several sectors.

The strongest sector-to-market edges are:

| Resolution | Edge | Total Conditional Score | Core Causal Score | Evidence |
|---|---|---:|---:|---|
| Daily | `XLY -> ^GSPC` | 3 | 2 | CVAR, PCMCI, XGBoost |
| Daily | `XLK -> ^GSPC` | 2 | 2 | CVAR, PCMCI |
| Weekly | `XLV -> ^GSPC` | 2 | 1 | PCMCI, XGBoost |

The strongest market-to-sector edges are:

| Resolution | Edge | Total Conditional Score | Core Causal Score | Evidence |
|---|---|---:|---:|---|
| Daily | `^GSPC -> XLE` | 2 | 2 | CVAR, PCMCI |
| Daily | `^GSPC -> XLB` | 2 | 2 | CVAR, PCMCI |
| Daily | `^GSPC -> XLK` | 2 | 1 | PCMCI, XGBoost |
| Daily | `^GSPC -> XLP` | 2 | 1 | PCMCI, XGBoost |
| Weekly | `^GSPC -> XLK` | 2 | 1 | PCMCI, XGBoost |
| Daily | `^GSPC -> XLF` | 2 | 1 | PCMCI, XGBoost |

This two-way structure is realistic. The S&P 500 is a broad aggregate, so market-wide shocks can affect all sectors. At the same time, some sectors may react first to information that later appears in the index. The results therefore do not suggest a simple one-directional system. Instead, they suggest a feedback network where Consumer Discretionary and Technology contain especially useful short-run information about the market.

---

## 5. Discussion

The main finding is that Consumer Discretionary is the strongest first-mover sector in this project. It has the highest overall source-sector support across the network and the strongest edge into the S&P 500. This is important because it suggests that the market may respond quickly to information about consumer demand and cyclical growth expectations.

One interpretation is that discretionary stocks are a forward-looking signal of household economic strength. If investors expect stronger consumer spending, lower recession risk, or easier credit conditions, Consumer Discretionary firms may benefit earlier than the broad index. Conversely, weakness in discretionary stocks may signal deteriorating expectations before the broader market fully adjusts.

Technology is also an important leader, but its role is slightly different. Technology leadership may reflect the sector's large index weight and its importance in growth expectations. Since Technology is a major component of the S&P 500, movements in `XLK` can mechanically and informationally relate to future index movements. The fact that `XLK -> ^GSPC` is supported by both conditional statistical methods makes it a meaningful result, but `XLK` is less dominant than `XLY` in the overall network ranking.

The frequency comparison is also central to the interpretation. Daily data produced the most evidence, while monthly data produced almost no conditional causal structure. This suggests that the relationships are short-horizon effects rather than slow-moving monthly trends. In practical terms, sector lead-lag signals may be more useful for understanding near-term market dynamics than for predicting long-term market direction.

Another important point is that the results are not the same as structural causation. The project does not prove that changes in Consumer Discretionary returns directly cause changes in the S&P 500. Instead, the evidence shows that lagged `XLY` returns improve prediction of future `^GSPC` returns within the observed system. The true mechanism could involve omitted macroeconomic variables, common reactions to news, liquidity differences, or investor rebalancing behavior.

---

## 6. Limitations

Several limitations should be considered when interpreting the results.

First, the analysis only includes sector ETFs and the S&P 500. Important macroeconomic variables such as interest rates, inflation expectations, unemployment news, volatility, credit spreads, and Federal Reserve announcements are not included. These omitted variables could create apparent lead-lag relationships.

Second, the selected lag is 1 for all resolutions. This makes the results easy to compare, but some relationships may operate over longer horizons. A sector might lead the market over two or three days even if the one-day lag is the strongest BIC-selected specification.

Third, XGBoost feature importance is not a causal test. It is useful as supporting predictive evidence, but it should not be interpreted in the same way as conditional VAR or PCMCI.

Fourth, the monthly sample is much smaller than the daily sample. With only 330 monthly observations, the monthly analysis has less statistical power, so the absence of strong monthly results does not necessarily mean that no monthly relationships exist.

Finally, financial relationships are not stable forever. Sector leadership can change across market regimes, especially during recessions, inflation shocks, policy changes, or sector-specific bubbles.

---

## 7. Conclusion

This project finds that the most consistent sector leader for the S&P 500 is **Consumer Discretionary (`XLY`)**. The daily edge `XLY -> ^GSPC` is supported by conditional VAR Granger causality, PCMCI, and XGBoost feature importance, making it the strongest individual result in the analysis.

The second strongest sector is **Technology (`XLK`)**, which is supported by both conditional statistical methods at the daily frequency. Healthcare (`XLV`), Materials (`XLB`), and Energy (`XLE`) provide additional predictive signals, but their evidence is weaker or less causal in nature.

The main conclusion is that sector leadership exists most clearly at the daily level. Consumer Discretionary appears to act as an early signal of broader market movement, likely because it is closely connected to expectations about consumer spending, growth, and risk appetite. However, the results should be interpreted as predictive temporal precedence rather than definitive proof of structural causation.

In short, the best answer to the project question is:

> **Consumer Discretionary (`XLY`) is the clearest first-mover sector and the strongest sector-level leading indicator for the S&P 500 in this dataset.**

---

## Appendix: Key Project Files

- `sector_data_all.csv`
- `phase3_outputs/tables/adf_stationarity_results.csv`
- `phase3_outputs/tables/var_lag_selection_results.csv`
- `phase3_outputs/tables/conditional_var_granger_fdr_adjusted_results.csv`
- `phase4_outputs/tables/consensus_edges.csv`
- `phase4_outputs/tables/strongest_consensus_edges.csv`
- `phase4_outputs/tables/sp500_overall_sector_leader_ranking.csv`
- `phase4_outputs/tables/xgboost_top_edges.csv`
- `phase4_outputs/tables/xgboost_model_performance.csv`
- `phase4_outputs/tables/pcmci_significant_edges.csv`
