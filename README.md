# CTR / Engagement Opportunity Scoring
### A FlyRank ML Internship Capstone — Andrew Adel

---

## Abstract

This paper addresses the problem of identifying high-visibility search content that under-captures organic clicks and user engagement. Using the FlyRank internship dataset aggregated to over 788,000 monthly content records, we engineered time-series search and behavioral features across GSC and GA4 metrics. A Random Forest classifier was trained on historical performance windows using a time-aware split to predict next-month underperformance without temporal data leakage. Benchmarked against a transparent baseline heuristic on the same holdout month, the Random Forest model achieved higher precision (0.78 vs 0.69) — meaning its top-ranked flags were more reliable — while the baseline heuristic achieved higher recall, F1-score, and ROC-AUC, indicating it caught a broader share of genuinely underperforming pages; this is a real precision/recall trade-off rather than a uniform win for either method. The resulting pipeline generates a prioritized content queue complete with actionable reason codes to guide automated content maintenance and metadata optimization.

---

## 1. Introduction / Problem Statement

Pages that rank well in organic search but fail to convert that visibility into clicks or on-site engagement represent a hidden opportunity: they already have the hardest part (rankings) solved, yet underperform on the part that actually drives business value.

- **Decision Supported:** Automated prioritization of content assets requiring metadata refresh, hook rewriting, or structural SEO updates.
- **Unit of Analysis:** A content asset (`content_hash_id`), aggregated at the monthly performance level.
- **Model Output:** A continuous opportunity-probability score, an ordinal rank, and an assigned reason code (e.g. `CTR_UNDERPERFORM_HIGH_POSITION`, `LOW_GA4_ENGAGEMENT`).
- **Human Action:** SEO editors and content teams review the top-ranked assets to update title tags, meta descriptions, and content headers.
- **Cost of Being Wrong:** False positives waste editor time on already-healthy pages; false negatives leave high-potential pages underperforming in organic search indefinitely.
- **Why Machine Learning:** Rule-based heuristics struggle to capture non-linear interactions between search impressions, position decay, and behavioral engagement metrics — motivating a learned model as a complement (not a replacement) to a simple heuristic.

---

## 2. Data

- **Source:** FlyRank Internship Data Warehouse (`fact_content_daily_performance`), accessed via a gated Hugging Face dataset and aggregated with DuckDB.
- **Date Window:** All months present in the warehouse slice used for this project (see the notebook's data-loading cell for the exact printed range).
- **Exclusions:** Records with fewer than 100 total monthly impressions were dropped to stabilize click-through-rate estimates and remove low-volume noise.
- **Leakage Controls:** Target-derived fields (`next_ctr`, `next_position`) were strictly excluded from the feature matrix; all features are built from month *t* to predict month *t+1*.
- **Privacy:** All client and content identifiers are pre-hashed by the warehouse. No client names, domains, raw URLs, or search queries appear anywhere in this project or repository.
- **Credential Handling:** No access token is hardcoded anywhere in the codebase. The Hugging Face token is read at runtime from an `HF_TOKEN` environment variable.

---

## 3. Methodology

**Baseline heuristic:**

$$\text{Baseline Score} = \frac{1}{\text{mean\_position}} \times (1 - \text{realized\_ctr})$$

The baseline is evaluated on the *exact same* holdout month, using the *exact same* underperformance definition as the ML model — this is what makes the head-to-head comparison in Section 4 fair rather than apples-to-oranges.

**Target label:** A content asset is labeled "underperforming" for month *t+1* if `next_position <= 5.0` **and** `next_ctr < 0.05`.

**Features:** `total_impressions`, `realized_ctr`, `mean_position`, `total_sessions`, `ga4_engagement_rate`, `total_ai_sessions`.

**Validation design:** A chronological, time-aware train/test split — every month except the most recent forms the training set; the single most recent month is held out untouched as the test set. This avoids the data leakage that a random/shuffled split would introduce in a time-series setting.

**Model:** `RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)`, chosen for its ability to capture non-linear feature interactions while remaining interpretable via feature importances.

---

## 4. Results

| Approach | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|
| Baseline Heuristic | 0.687 | **0.424** | **0.524** | **0.875** |
| Random Forest ML | **0.781** | 0.272 | 0.404 | 0.867 |

**Head-to-head:** the Random Forest model achieved higher Precision (0.78 vs 0.69) but lower Recall, F1-Score, and ROC-AUC than the baseline heuristic. In practice, this means the ML model's top-ranked flags are more reliable, but it misses a larger share of pages that genuinely go on to underperform — which the simpler heuristic catches.

**Key feature drivers:** `mean_position` and `realized_ctr` are the primary predictive drivers, followed by `total_impressions` and `ga4_engagement_rate` (see `outputs/feature_importance.png` in the repo for the full chart).

**Why the baseline held up:** the Random Forest's discrimination (ROC-AUC 0.87) is close to, and slightly below, the baseline heuristic's (0.88). This suggests the heuristic's simple formula already captures most of the usable signal in position and CTR, and further ML gains would likely require additional engineered features (e.g. multi-month trend/momentum signals) rather than model complexity alone.

---

## 5. Limitations & Honest Framing

Results represent **directional, decision-support signals**, not proven causal relationships. CTR fluctuations may also stem from SERP feature changes (e.g. Knowledge Panels, AI Overviews) rather than content quality gaps. This project does not claim to reverse-engineer any search engine's ranking algorithm, nor does it guarantee ranking or CTR improvements from acting on its recommendations.

---

## 6. Ranked Recommendations

Content assets are prioritized by predicted opportunity score and exported as a ranked action queue with reason codes and suggested interventions:

| Reason Code | Trigger | Suggested Action |
|---|---|---|
| `CTR_UNDERPERFORM_HIGH_POSITION` | High rank but low CTR | `REFRESH_METADATA_OR_SNIPPET` |
| `LOW_GA4_ENGAGEMENT` | High rank but low engagement | `REWRITE_CONTENT_HOOKS` |

**Confidence & limits:** given the precision/recall trade-off in Section 4, treat this queue as a high-precision, conservative shortlist — some genuinely underperforming pages may not surface near the top. Recommendations are decision-support signals, not algorithmic guarantees.

---

## 7. Reproducibility

- **Environment:** `pip install duckdb pandas scikit-learn matplotlib numpy`
- **Execution:** Run `notebooks/capstone.ipynb` (or `work/notebooks/capstone.ipynb`, depending on repo layout) top to bottom.
- **Random Seed:** Fixed at `42` across all model training steps.
- **Data Token:** Configured via a DuckDB Hugging Face secret, read from the `HF_TOKEN` environment variable at runtime — never hardcoded in the notebook or committed to the repo.
- **Full source:** [github.com/Andrew-adel391/flyrank-ml-internship](https://github.com/Andrew-adel391/flyrank-ml-internship) — all weekly assignment notebooks and this capstone notebook.

---

## 8. Acknowledgments & Data Credit  

This work uses data provided by [FlyRank](https://flyrank.ai).
