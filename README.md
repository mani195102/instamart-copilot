# Instamart Customer Intelligence & Support Copilot

## Overview

An internal Copilot for the (fictional) Swiggy Instamart customer-experience team, built in four connected Parts on fully synthetic data. All amounts are in Indian Rupees (INR).

| Part | What it does | Main script(s) | Feeds |
|---|---|---|---|
| A | Audits the dirty raw exports, cleans them with documented rules, stores them as CSV + SQLite, answers 7 SQL questions and cross-checks them in pandas | `src/part_a_clean.py`, `src/part_a_validate.py`, `src/part_a_analysis.py`, `sql/queries.sql` | B (clean CSVs), D (`data/instamart.db`) |
| B | RFM segmentation with K-means (K=4, named segments, Premium flag) and a leakage-free lapse model | `src/part_b_segments.py`, `src/part_b_lapse.py` | C (`outputs/customer_segments.csv`) |
| C | Support-ticket triage: input guardrails, PII redaction, zero/few-shot prompts, retries, output validation, token budget, segment-aware routing | `src/part_c_triage.py` | — |
| D | Policy RAG assistant (40-word overlapping chunks, TF-IDF retrieval, grounded answers) and a ReAct refund agent with tools over `data/instamart.db`, iteration/token/time limits, a refund ledger and checkpoint-resume | `src/policy_rag.py`, `src/part_d_agent.py` | — |

Every language-model call goes through the supplied offline `mock_llm.py`; no key, network or paid service is used. `generate_data.py` and `mock_llm.py` are committed unmodified. Libraries: Python standard library, pandas, numpy, scikit-learn, joblib (scipy is installed as a scikit-learn dependency). Developed with Python 3.14.0 and SQLite 3.50.4.

Repository layout:

```
generate_data.py  mock_llm.py  requirements.txt  .gitignore  README.md
sql/queries.sql
src/part_a_clean.py  part_a_validate.py  part_a_analysis.py
src/part_b_segments.py  part_b_lapse.py
src/part_c_triage.py
src/policy_rag.py  part_d_agent.py
data/raw/  data/policies/  data/clean/  data/instamart.db        (generated)
outputs/  state/                                                  (generated)
```

## Reproduce

Run from the repository root. On Windows use `.venv\Scripts\activate` in step 2 instead of `source .venv/bin/activate`.

1. `python -m venv .venv`
2. `source .venv/bin/activate`
3. `pip install -r requirements.txt`
4. `python generate_data.py` → last line `generated: customers=1200 products=40 orders_raw=13084 tickets=164`
5. `python src/part_a_clean.py` → audit tables, cleaning log, `data/clean/*.csv`, `data/instamart.db`
6. `python src/part_a_validate.py` → `ALL CHECKPOINTS PASSED`
7. `python src/part_a_analysis.py` → Q1–Q7, pandas cross-check, statistics
8. `python src/part_b_segments.py` → K-means table, segment profile, PCA, `outputs/customer_segments.csv`
9. `python src/part_b_lapse.py` → grid searches, test metrics, `outputs/lapse_model.joblib`, `outputs/retention_list.csv`
10. `python src/part_c_triage.py` → retry log, zero/few-shot comparison, `outputs/triage_results.csv`, `outputs/triage_budget_demo.csv`
11. `python src/policy_rag.py` → chunk counts, retrieval check, policy QA evaluation with overlap 10 and overlap 0
12. `python src/part_d_agent.py` → seven scenarios, stop-safely demos, crash-and-resume, `outputs/agent_runs.jsonl`, `outputs/refund_ledger.jsonl`, `state/`

Each script is deterministic (fixed seeds: `random_state=42` everywhere, `random.Random(42)` for retry jitter). Steps 10 and 12 delete and rebuild their own outputs, so they can be re-run safely.

## Part A

### Audit of the raw data

Raw files are read with `keep_default_na=False, na_values=[""]`, so only truly blank cells count as missing and the text `N/A` stays text for the audit (pandas' default would silently turn `N/A` into missing and hide 53 of the 120 bad amounts). `tickets.csv` is read with `keep_default_na=False`.

| Table | Shape | Column: dtype, missing (count, %) |
|---|---|---|
| customers.csv | 1,200 × 5 | customer_id str 0 (0%); city str 0 (0%); signup_date str 0 (0%); age float 61 (5.08%); gender str 30 (2.50%) |
| products.csv | 40 × 4 | product_id, product_name, category str, price_inr int; 0 missing in every column |
| orders.csv | 13,084 × 9 | order_id, customer_id, product_id, order_ts, status, order_amount str; quantity, discount_pct int; delivery_minutes float. Missing: delivery_minutes 2,541 (19.42%), every other column 0 (0%) |
| tickets.csv | 164 × 5 | all str; 4 empty texts kept as empty strings |

| Audit figure | Value |
|---|---|
| raw orders.csv rows | 13,084 |
| exact duplicate rows | 119 |
| rows with status pending | 656 |
| distinct order_id | 12,309 |
| order_amount not convertible to a number (`N/A`, `error`) | 120 |
| negative delivery_minutes | 106 |
| blank delivery_minutes | 2,541 |
| blank age | 61 |
| age outside 10–100 | 6 |
| distinct raw city spellings | 24 |
| distinct non-blank raw gender spellings | 10 |
| blank gender | 30 |

### Cleaning rules (with the number of values changed)

Customers:

| # | Rule | Changed |
|---|---|---|
| C1 | `city` → strip spaces, title case (24 spellings → 6 cities) | 476 values |
| C2 | `gender` → `male`/`female` (from male, Male, M, m, MALE / female, Female, F, f, FEMALE); blank → `unknown` | 950 values changed (30 → unknown) |
| C3 | ages outside 10–100 → missing, then every missing age ← median age of the customer's city (`groupby("city")["age"].transform("median")`) | 6 set to missing; 67 filled (61 blank + 6 impossible) |
| C4 | `signup_date` → datetime | 1,200 |

Result: Pune 227, Delhi 210, Bengaluru 197, Chennai 191, Hyderabad 191, Mumbai 184; female 610, male 560, unknown 30; 0 missing ages, all ages 18–60. City medians used: Bengaluru 42, Chennai 39, Delhi 43, Hyderabad 37.5, Mumbai 38.5, Pune 39.

Orders (applied in this order):

| # | Rule | Changed |
|---|---|---|
| O1 | `order_ts` → datetime | 13,084 |
| O2 | drop exact duplicate rows | 119 rows removed |
| O3 | per `order_id` keep only the record with the latest `order_ts` (removes the earlier `pending` record) | 656 rows removed |
| O4 | `order_amount` → numeric (`errors="coerce"`); missing ← `round(price_inr × quantity × (1 − discount_pct/100), 2)` | 112 amounts filled (the other 8 bad amounts sat on removed duplicate/pending rows) |
| O5 | negative `delivery_minutes` → missing | 105 (the 106th sat on a removed exact duplicate) |
| O6 | completed orders: missing `delivery_minutes` ← median of the product category among completed orders (`groupby` + `transform`); cancelled orders stay missing because they were never delivered | 774 filled; medians Dairy 25, Electronics 25, Grocery 25, Personal Care 25.5, Snacks 25 |

Result: 12,309 rows, one per order_id; 11,110 completed, 1,199 cancelled, 0 pending; 0 missing order_amount; 0 missing delivery_minutes among completed and 1,199 among cancelled orders; completed revenue INR 2,998,668.80.

### Store and validate

Clean tables are written to `data/clean/{customers,products,orders}.csv` and to `data/instamart.db` (tables `customers`, `products`, `orders`; `order_ts` stored as text `YYYY-MM-DD HH:MM:SS`). `src/part_a_validate.py` asserts 17 checkpoints (city and gender counts, ages, row counts, statuses, missing values, revenue, the DB tables and the timestamp format) and prints `ALL CHECKPOINTS PASSED`.

### SQL analysis (`sql/queries.sql`, run with `pd.read_sql_query`)

| Query | Technique | Result |
|---|---|---|
| Q1 | WHERE (completed, amount > 500) then HAVING total > 8,000 | 30 customers (top: C0693, INR 13,776.55 over 6 orders) |
| Q2 | 3-table join, revenue by city × category | 30 rows, sum INR 2,998,668.80; largest Bengaluru / Electronics INR 233,909.45 |
| Q3 | scalar subquery, price > average | 11 products |
| Q4 | LEFT JOIN … IS NULL | 25 customers with no completed order |
| Q5 | two CTEs | 550 customers above the average of INR 2,552.06 |
| Q6 | RANK / DENSE_RANK / ROW_NUMBER by city | top: Bengaluru C0082 (12,856.20), Chennai C0930 (10,151.40), Delhi C0291 (13,742.60), Hyderabad C0743 (12,730.05), Mumbai C0693 (14,766.55), Pune C0426 (10,377.10); 8 rows have RANK() ≠ ROW_NUMBER() |
| Q7 | running total `SUM(revenue) OVER (ORDER BY day)` | 183 daily rows, last cumulative INR 2,998,668.80 |

Q6 ties: when two customers in a city have exactly the same total spend, RANK() gives them the same rank but ROW_NUMBER() still numbers them consecutively, so the second of the pair, and anyone ranked after a tie, shows RANK() ≠ ROW_NUMBER(). The 8 rows: Bengaluru C0734, C0723; Mumbai C1053, C0608; Pune C1064, C0057, C0936, C0276.

### pandas cross-check and exploratory analysis

`pd.pivot_table(..., margins=True)` over completed orders reproduces Q2; its grand total is INR 2,998,668.80, equal to the SQL total.

| City | Sum (INR) | Mean (INR) | Count |
|---|---|---|---|
| Hyderabad | 528,589.90 | 284.95 | 1,855 |
| Bengaluru | 523,643.40 | 276.47 | 1,894 |
| Delhi | 502,829.80 | 243.62 | 2,064 |
| Pune | 500,918.05 | 252.48 | 1,984 |
| Mumbai | 479,058.50 | 297.00 | 1,613 |
| Chennai | 463,629.15 | 272.72 | 1,700 |

Completed `order_amount` (INR): mean 269.91, median 135.00, skewness 4.61, min 12.00, Q1 64.80, Q3 297.00, IQR 232.20, max 3,998.00. Fences: lower −283.50 (0 outliers below), upper 645.30 (897 outliers above).

Correlations over completed orders: amount–discount −0.095, amount–quantity +0.087, delivery_minutes with anything ≤ 0.013 in absolute value.

- **What is the data about?** About 12,300 quick-commerce orders from 1,175 ordering customers in six cities, April–September 2025, across 40 products in five categories.
- **Distributions?** Order value is strongly right-skewed: the mean (INR 269.91) is double the median (INR 135.00), with skewness 4.61. Most baskets are small groceries, and a long tail of electronics pulls the mean up. Delivery times centre on about 25 minutes.
- **Outliers?** 897 orders sit above the INR 645.30 upper fence and none below. These are genuine electronics purchases, not errors, so they are kept. The real anomalies (120 unparseable amounts, 106 negative times, duplicates) were fixed in cleaning.
- **Relationships?** All correlations are weak. The largest in absolute value is order_amount with discount_pct (−0.095), just ahead of order_amount with quantity (+0.087). A correlation this small explains under 1% of the variance and does not establish causation; for example, discounts are not shown to lower basket value.

## Part B

### RFM (snapshot 2025-10-01 00:00:00, completed orders)

1,175 customers. Recency median 19, min 0, max 180 days; frequency mean 9.46, max 33; monetary mean INR 2,552.06, max INR 14,766.55.

### K-means scan (StandardScaler on R, F, M; `n_init=10, random_state=42`)

| K | Inertia | Silhouette | Inertia drop vs K−1 |
|---|---|---|---|
| 2 | 1820.8 | 0.465 | — |
| 3 | 1268.2 | 0.445 | 30.3% |
| 4 | 743.1 | **0.526** | **41.4%** |
| 5 | 617.4 | 0.480 | 16.9% |
| 6 | 503.2 | 0.482 | 18.5% |
| 7 | 408.5 | 0.416 | 18.8% |
| 8 | 366.5 | 0.412 | 10.3% |

**Chosen K = 4.** The elbow is at K=4: the 3→4 step removes 41% of inertia and every later step removes under 19%. K=4 also has the highest silhouette (0.526), and its four groups map onto four distinct campaigns.

Inertia always falls as K grows: more centroids can only bring points closer to their nearest centre, and at K = n it reaches 0. A lower inertia for K=8 than for K=2 is therefore guaranteed by construction and says nothing about whether the extra clusters are real. K has to be chosen from the diminishing returns (elbow), cluster separation (silhouette) and whether the business can act on the groups.

### Segments (K=4; cluster ids 0–3 are arbitrary labels with no order)

| segment_id | Name | Size | Mean recency (days) | Mean frequency | Mean monetary (INR) | Premium | Recommended action |
|---|---|---|---|---|---|---|---|
| 1 | Premium Big Spenders | 138 | 24.1 | 6.77 | 7,129.70 | **yes** | Priority 1-hour support, early access to electronics launches, no-cost EMI offers |
| 2 | Loyal Regulars | 419 | 8.8 | 19.40 | 3,410.02 | no | Membership with free delivery; reward streaks instead of discounting |
| 0 | Occasional Shoppers | 436 | 28.9 | 3.77 | 1,057.62 | no | Cross-sell dairy/snack bundles and basket-size thresholds to lift frequency |
| 3 | Dormant / At-Risk | 182 | 109.0 | 2.20 | 685.99 | no | Time-limited win-back coupon on staples; stop after two ignored offers |

Names are assigned in code from the profile, never from the cluster number: highest mean monetary → Premium, highest recency → Dormant, highest frequency → Loyal.

PCA(2) on the scaled RFM matrix: explained_variance_ratio_ = 0.634, 0.218 (total 0.852). PC1 is an "engagement" axis (recency −0.58, frequency +0.61, monetary +0.53); PC2 mostly separates high spend from high frequency (monetary +0.82).

Output: `outputs/customer_segments.csv` (customer_id, recency, frequency, monetary, segment_id, segment_name, is_premium), with 138 premium customers.

### Lapse problem

Cutoff 2025-09-01. The cohort is every customer with a completed order before the cutoff: **1,156 customers, 443 lapsed (38.32%)**, where lapsed means no completed order from 2025-09-01 to 2025-09-30. Split: 924 train / 232 test (stratified, `random_state=42`).

Features, built only from orders dated before 2025-09-01 plus the clean customer table:

- numeric: recency, frequency, monetary, avg_discount_pct, cancel_rate (cancelled ÷ completed+cancelled), avg_delivery_minutes, age, tenure_days, plus two extras: avg_order_value and distinct_categories
- nominal: city, gender, fav_category (ties broken alphabetically)

**Leakage controls.** No order dated 2025-09-01 or later enters any feature; those orders are used only to build the label. The segment labels from Part B task 3 are not used as features. No encoder, scaler or resampler saw test rows: the `OneHotEncoder(drop="first")` (k−1 columns) and `StandardScaler` live inside a scikit-learn `Pipeline` fitted on training rows only, and over-sampling touches training rows only. The code asserts 0 missing values in all feature columns.

The scaler's learned mean of frequency is 8.1147 on the training rows versus 8.0285 on the full cohort. They differ because the scaler only ever saw the 924 training customers, and that is exactly the training-only rule.

### Models (scoring = F1 of the lapsed class, cv=5)

| Model | Grid | Best params | Best CV F1 |
|---|---|---|---|
| LogisticRegression(max_iter=1000) | C ∈ {0.01, 0.1, 1, 10, 100} | C = 1 | 0.652 |
| RandomForest(n_estimators=100, random_state=42) | max_depth ∈ {3,5,8} × min_samples_leaf ∈ {5,10,20} (45 fits) | max_depth 5, min_samples_leaf 20 | 0.686 |

Test set (232 rows; lapsed = positive class):

| Model | Confusion [[TN FP] [FN TP]] | Precision | Recall | F1 | Accuracy |
|---|---|---|---|---|---|
| Baseline "never lapsed" | [[143 0] [89 0]] | 0.000 | 0.000 | 0.000 | 61.6% |
| Logistic regression | [[107 36] [32 57]] | 0.613 | 0.640 | 0.626 | 70.7% |
| **Random forest (selected)** | [[110 33] [30 59]] | 0.641 | 0.663 | **0.652** | 72.8% |

**Choice: random forest.** It has the best F1 (0.652 vs 0.626) and wins on both precision and recall, so there is no trade-off to argue over. A missed lapser is costly, because it is a customer who walks away with no win-back attempt and the lost future orders are worth far more than one coupon. A wrongly targeted loyal customer only costs a discount they partly would have spent anyway. The forest catches 59 of 89 lapsers (2 more than logistic regression) while also sending 3 fewer wasted offers. Logistic regression stays useful as the explainable cross-check below.

(With only the required features the forest picks depth 8 / leaf 10 with test F1 0.655 and logistic regression picks C=100 with F1 0.648. The two extra features change the chosen hyper-parameters, not the order of magnitude.)

### Class-balance experiment (logistic regression, C=1)

Lapsed training rows were over-sampled with replacement (`sklearn.utils.resample`, `random_state=42`) from 354 to 570, matching the 570 non-lapsed rows. The test set was not changed.

| Training data | Precision | Recall | F1 |
|---|---|---|---|
| original (354 / 570) | 0.613 | 0.640 | 0.626 |
| balanced (570 / 570) | 0.592 | **0.865** | 0.703 |

Recall moved most, from 0.64 to 0.865, while precision stayed near 0.6 (it fell slightly). For the retention team, the balanced model finds 77 of 89 lapsers instead of 57, at the price of 53 instead of 36 offers sent to customers who would have stayed. That trade is worth it when a win-back offer is cheap relative to a lost customer.

### Interpretation

Random-forest top-5 importances:

| Feature | Importance | Business reading |
|---|---|---|
| frequency | 0.313 | How often someone orders is the strongest signal of habit |
| monetary | 0.260 | Spend level separates low-touch, high-ticket buyers from daily-staples buyers |
| recency | 0.083 | A long gap since the last order is an early warning |
| cancel_rate | 0.075 | Customers who cancel more are less settled in the habit |
| distinct_categories | 0.063 | Customers who buy across more categories are more embedded |

Logistic regression signs (standardised features):

- frequency **−1.93**: every extra order before the cutoff makes lapsing much less likely, so build habit with streak rewards.
- recency **+0.11**: the longer since the last order, the more likely the customer has gone, so trigger outreach on days-since-last-order.
- monetary **+0.39**: at equal frequency, higher spend raises lapse risk. These are infrequent high-ticket (electronics) buyers, so don't treat high spend as safe.

Persistence: `joblib.dump` to `outputs/lapse_model.joblib`; after `joblib.load` the reloaded model reproduces all 232 test predictions exactly (asserted in code).

### Retention list (`outputs/retention_list.csv`)

40 test-set customers are Premium. The top 10 by predicted lapse probability (0.51 down to 0.45) are written with customer_id, lapse_probability, actually_lapsed. **0 of these 10 actually lapsed.** All their probabilities are near 0.5, and premium customers rarely lapse in this data, so the list is mostly cautious early warnings.

The Premium flag comes from the whole-window RFM, which includes the September label window, so it is a post-hoc whole-window label. The retention list is therefore illustrative and not a leakage-free evaluation. The model's own features still use only pre-cutoff orders.

## Part C

Tickets are processed in file order. Each user message contains exactly the redacted ticket text and nothing else.

### Input guardrails and redaction

| Check | Count |
|---|---|
| empty / whitespace-only → `skipped_empty` (never sent) | 4 |
| `est_tokens(text) > 400` → `flagged_long` (never sent) | 4 (each estimates to 734 tokens) |
| eligible tickets sent to the model | 156 |

| Placeholder | Pattern | Substitutions |
|---|---|---|
| [EMAIL] | `[\w.+-]+@[\w-]+\.[\w.]+` | 55 |
| [PHONE] | `\b[6-9]\d{9}\b` | 91 |
| [ORDER_ID] | `ORD-\d{6}` | 132 |
| **Total** | applied in this order | **278** |

Re-applying all three patterns to the redacted texts finds 0 matches (asserted in code).

### Prompts

Both system prompts contain `CLASSIFY`, list billing, delivery, technical and fraud, and demand a JSON-only reply `{"category": "...", "reason": "..."}`. The zero-shot prompt has no `EXAMPLES:`. The few-shot prompt adds an `EXAMPLES:` line followed by four labelled examples (fraud, technical, billing, delivery). Both are in `src/part_c_triage.py`.

### Resilience and validation

Every call goes through `call_with_retry`: at most 3 retries, wait before retry n = `0.05 × 2**n × uniform(0.9, 1.15)` s, with one log line per retry, for example `[retry] ticket=T0002 retry=0 wait=0.0530s error=503 server busy`. `mock_llm.reset()` is called before every run. Each full run hit **19 TransientLLMErrors, all recovered on the first retry; 0 permanent failures.** With this mock every retry is retry 0, and the logged waits run from 0.0451 to 0.0574 s, inside the 0.045–0.0575 s range. Before every call the code asserts that the user message is exactly the redacted text and that none of the three PII patterns matches it.

Each reply is validated: valid JSON with an allowed category gives `classified_json`. Otherwise the words from `re.findall(r"[a-z]+", text.lower())` give `classified_keyword` only if exactly one allowed category word appears. Everything else, including JSON with category `shipping`, gives `flagged_invalid_output` and goes to human review without a guess.

| Status (per full run) | Tickets |
|---|---|
| classified_json | 135 |
| classified_keyword | 12 |
| flagged_invalid_output | 9 |
| skipped_empty | 4 |
| flagged_long | 4 |

### Zero-shot vs few-shot (no budget)

| Prompt | Classified | Correct | Accuracy | Total tokens | Tokens / classified |
|---|---|---|---|---|---|
| zero-shot | 147 | 126 | 85.7% | 13,768 | 93.7 |
| few-shot | 147 | 147 | 100.0% | 29,680 | 201.9 |

Zero-shot confusion (rows = true, columns = predicted):

| true \ pred | billing | delivery | fraud | technical |
|---|---|---|---|---|
| billing | 38 | 0 | 0 | 0 |
| delivery | 0 | 36 | 0 | 0 |
| fraud | **11** | 0 | 23 | 0 |
| technical | **10** | 0 | 0 | 29 |

The few-shot confusion table is perfectly diagonal (38 / 36 / 34 / 39).

All 21 zero-shot errors are fraud or technical tickets mislabelled as billing, because they also mention "charged" or "payment". Eleven of those are fraud tickets that would miss the security team. The few-shot system prompt is 162 estimated tokens against 60 for zero-shot. That raises the cost from 93.7 to 201.9 tokens per classified ticket (29,680 vs 13,768 in total, +15,912), and it fixes all 21 errors (100% vs 85.7%). The extra ~108 tokens per ticket are worth paying whenever a misroute is expensive, as with a fraud report landing in the billing queue. Zero-shot is only good enough for a cheap first pass whose billing labels are re-checked anyway.

### Token budget demo (`outputs/triage_budget_demo.csv`)

Before each model call, if the running total of `usage.total_tokens` is already ≥ `TOKEN_BUDGET`, no call is made and the ticket gets `budget_exhausted` (routed to human review). The zero-shot pipeline with `TOKEN_BUDGET = 6000` makes 68 calls (6,008 tokens; the last call starts below the budget and finishes just over it). **88 tickets end `budget_exhausted`**, the first at file row 71 (T0051). Before the budget ran out: 59 classified_json, 5 classified_keyword, 4 flagged_invalid_output, plus the 8 empty/long tickets caught by the input guardrails. The file has budget, tokens_used and cumulative_tokens per ticket.

### Routing and final output (`outputs/triage_results.csv`)

The final file comes from a few-shot run with `TOKEN_BUDGET = 100000`, which never binds: 29,680 tokens, 156 calls, 135 / 12 / 9. Each ticket is joined to `outputs/customer_segments.csv`; a customer with no row counts as not premium (all 164 tickets here have a row).

| Rule (checked in order) | Route | Priority | Tickets |
|---|---|---|---|
| status skipped_empty, flagged_long, flagged_invalid_output or budget_exhausted | human_review | NORMAL | 17 (4 + 4 + 9 + 0) |
| category fraud | security_team | HIGH | 34 |
| customer is_premium | premium_desk | HIGH | 11 |
| otherwise | support_queue | NORMAL | 102 |

The file has 164 rows with columns ticket_id, customer_id, status, category, segment_name, priority, route, tokens_used. Category is blank when there is no valid answer.

**What was rule-validated and what went to humans.** 147 tickets were classified by the model and routed automatically. Their outputs passed the validation rules: 135 as valid JSON with an allowed category, 12 through the single-category-word rule. 17 tickets were never classified and went to human review: 4 empty and 4 over-long tickets that were never sent to the model, and 9 replies that failed validation (a category outside the list) and were not guessed.

## Part D

### Policy RAG assistant (`src/policy_rag.py`, tasks 1–4)

**Chunks.** Each of the 5 policy files is split into 40-word chunks with a 10-word overlap: each chunk starts 30 words after the previous one, stopping after the first chunk that reaches the last word. Every chunk records its source file. The result is **15 chunks**: refund 4, delivery 3, escalation 3, payment 3, privacy 2.

**Retrieval.** `TfidfVectorizer()` is fitted on the chunk texts, and the query is transformed with the same fitted vectorizer. Chunks are ranked by `cosine_similarity`, and `retrieve(query, k=3)` returns `[{source, score, text}]`. For `refund policy late delivery` the top three are all `refund_policy.txt` (scores 0.272, 0.251, 0.245).

TF (term frequency) measures how often a word occurs in a chunk, so a chunk that repeats "refund" scores higher for "refund". IDF (inverse document frequency) down-weights words that occur in many chunks, like "order" or "customer", so rare, specific words such as "wallet" or "OTP" decide the ranking. A limitation is that TF-IDF matches exact word forms only. "Money back" does not match "refund", and "how long until it is credited" shares almost no words with "within 5 working days", so a paraphrased question can retrieve the wrong chunk.

**Answering.** `answer(question)` retrieves 3 chunks and sends two messages. The system message contains `POLICY ASSISTANT` and tells the model to answer only from the context, or else to say it could not find the information. The user message has the form `CONTEXT:\n[<source>] <chunk>\n[<source>] <chunk>\n[<source>] <chunk>\nQUESTION:\n<question>`. It returns the model text and the three sources.

**Evaluation (`data/raw/policy_eval.csv`).**

| Metric | Overlap 10 (15 chunks) | Overlap 0 (13 chunks) |
|---|---|---|
| expected_doc among top-3 sources (12 answerable) | **12 / 12** | 12 / 12 |
| answer contains expected_fact, case-sensitive (12 answerable) | **12 / 12** | **10 / 12** |
| fallback "I could not find…" (2 unanswerable) | **2 / 2** | 2 / 2 |

With overlap 0, two questions break even though the right file is retrieved:

- *"What delivery time makes a completed order eligible for a full refund?"* gets the answer "receives a full refund of the order amount." The sentence "If a completed order is delivered in more than 45 minutes, the customer receives a full refund…" is cut by a chunk boundary, so the fact "more than 45 minutes" sits at the end of one chunk and the answer fragment at the start of the next.
- *"How soon is money returned when a payment fails but is deducted?"* gets an answer that stops at "…the amount is returned automatically", because "within 48 hours" falls into the next chunk.

The 10-word overlap keeps such sentences whole in at least one chunk.

### Refund agent (`src/part_d_agent.py`, tasks 5–8)

**Tools.** `TOOLS_SPEC` lists the five tools, each with a name, a one-sentence description and a JSON-schema parameter spec. `TOOL_MAP` maps each name to its Python function, and the loop calls `TOOL_MAP[name](**arguments)`.

| Tool | Behaviour |
|---|---|
| `get_order(order_id)` | reads `data/instamart.db` → `{order_id, status, order_amount, delivery_minutes}` or `{"error": "order not found"}` |
| `search_policy(query)` | `{"results": retrieve(query, k=3)}`, reusing the RAG retriever above |
| `calculate_refund(order_amount, delivery_minutes)` | `{"refund_inr": order_amount}` if delivered in more than 45 min, else `{"refund_inr": 0}` |
| `issue_refund(order_id, refund_inr)` | appends `{timestamp, tool, order_id, refund_inr, run_id}` to `outputs/refund_ledger.jsonl` → `{"refund_inr"}` |
| `escalate_to_human(order_id, refund_inr)` | same ledger line with tool `escalate_to_human` → `{"refund_inr"}` |

`issue_refund` also has two safety guards. It refuses amounts above INR 1000, which must be escalated. It is idempotent: it never writes a second payment for an order already in the ledger.

**ReAct loop.** The signature is `run_agent(user_message, scenario_id, max_iterations=6, token_limit=6000, time_limit_seconds=10)`. Messages start with a system message containing `AGENT` and the user message. Each tool request is appended as `{"role": "assistant", "content": None, "tool_calls": [...]}`, and the tool result as `{"role": "tool", "tool_call_id", "name", "content": <JSON string>}`. The function returns `status, final_answer, steps, llm_calls, tokens_used`, and its status is the word before the colon in the final answer. Every run appends one line to `outputs/agent_runs.jsonl` with scenario_id, status, final_answer, refund_inr, llm_calls, tokens_used and every step's tool, arguments and observation.

**Seven scenarios** (default limits):

| Scenario | Expected | Status | Refund (INR) | Model calls | Tokens used | Tools executed |
|---|---|---|---|---|---|---|
| S1 | REFUND 128.25 | REFUND | 128.25 | 5 | 1,077 | get_order, search_policy, calculate_refund, issue_refund |
| S2 | REFUND 135.00 | REFUND | 135.00 | 5 | 1,077 | get_order, search_policy, calculate_refund, issue_refund |
| S3 | ESCALATE 1999.00 | ESCALATE | 1,999.00 | 5 | 1,064 | get_order, search_policy, calculate_refund, escalate_to_human |
| S4 | DENIED | DENIED | 0.00 | 4 | 749 | get_order, search_policy, calculate_refund (delivered ≤ 45 min) |
| S5 | DENIED | DENIED | 0.00 | 3 | 445 | get_order, search_policy (order cancelled) |
| S6 | NOT_FOUND | NOT_FOUND | 0.00 | 2 | 144 | get_order (ORD-999999 does not exist) |
| S7 | FALLBACK | FALLBACK_MAX_ITERATIONS | 0.00 | 6 | 3,216 | search_policy × 6 |

All seven match the expected outcome and refund within INR 0.01. S7 completes all 6 model calls well inside the 6,000-token limit. The ledger holds issue_refund ORD-100048 (128.25), issue_refund ORD-100067 (135.00) and escalate_to_human ORD-100145 (1,999.00).

### Stop safely (task 7)

| Limit | Rule | Demonstration | Status |
|---|---|---|---|
| (a) iterations | after `max_iterations` model calls with no final answer | S7 with defaults | **FALLBACK_MAX_ITERATIONS** (6 calls) |
| (b) tokens | after each call, if `tokens_used > token_limit` stop at once, without running the requested tool | S1 with `token_limit=200` | **FALLBACK_TOKEN_LIMIT**: 3 calls, 462 tokens; get_order and search_policy done; the requested calculate_refund was never run |
| (c) time | at the top of each iteration, if `time.monotonic() - start >= time_limit_seconds` | S1 with `time_limit_seconds=0` | **FALLBACK_TIMEOUT**: 0 model calls |

Every fallback returns the steps completed so far and a hand-off summary in `final_answer`, for example: *"Handed to a human agent (token limit 200 exceeded). Request: 'My order ORD-100048 reached me very late. I would like a refund.'. Tools completed: get_order, search_policy. Model calls: 3, tokens: 462."*

### Crash and resume (task 8)

After every tool observation, the loop atomically writes `state/<run_id>.json`. The checkpoint holds the run id, scenario id, full message list, tokens_used, step number, tools executed, llm_calls and the limits. `resume_agent(run_id)` loads it and continues the same loop.

The demo runs S1 as run `S1_crash_resume` and raises a simulated crash right after the 2nd tool execution, once its checkpoint is saved. `resume_agent("S1_crash_resume")` then finishes the run.

| Phase | Tools executed, in order |
|---|---|
| before the crash | 1. get_order → 2. search_policy |
| after `resume_agent` | 3. calculate_refund → 4. issue_refund |

That is four executions with none repeated, and the final status is **REFUND, INR 128.25**. Because S1 had already been paid in the scenario run, the idempotent `issue_refund` returned the amount with a note and wrote no second payment. The ledger still holds exactly one refund for ORD-100048.

`state/` holds a checkpoint for each of the 7 scenarios plus `S1_token_limit_200`, `S1_time_limit_0` and `S1_crash_resume`.

### ReAct versus plan-and-execute

My ReAct agent makes one model call per tool plus one to answer. That measured 5 calls for S1–S3, 4 for S4, 3 for S5 (cancelled order) and 2 for S6 (unknown order). Each call resends the whole growing transcript, so S1 cost 1,077 tokens and the looping S7 cost 3,216.

A plan-and-execute agent would ask the model once for the full step list (get_order → search_policy → calculate_refund → issue_refund) and run the steps without the model. A happy path would then need about 2 calls (plan, final wording) and far fewer tokens, since tool outputs are not re-read at every step.

The difference shows on unexpected observations. With ReAct, a NOT_FOUND order (S6) or a cancelled order (S5) is noticed right after the observation, and the agent stops at 2–3 calls without running pointless tools. A fixed plan would run calculate_refund on a missing or cancelled order unless the executor detects the error and triggers a replan, which costs another call and needs extra error-handling code.
For this refund workflow I would choose ReAct. The path branches on data (status, minutes, the INR 1000 limit), the call counts are small (2–5), and every decision is grounded in the latest observation. The step, token and time limits already bound its cost. I would pair it with the deterministic tool-side guards (limit check, idempotent ledger) so that a wrong model step can never pay out money.

## Limitations and Next Steps

- **The retention list is illustrative.** The Premium flag is a post-hoc whole-window label: it comes from RFM over April–September, including the September window that defines the lapse label. The retention list is therefore illustrative, not a leakage-free evaluation. Next step: recompute segments on pre-cutoff orders only.
- **Synthetic data.** Every pattern (e.g. three customer archetypes, delivery ~N(24, 8)) comes from `generate_data.py`, so model scores and segment shapes will not transfer to real Instamart data.
- **Imputation choices.** City-median ages and category-median delivery times flatten variance. 774 completed orders carry an imputed time of about 25 minutes, so a missing time is assumed on time and can never trigger a late-delivery refund. A production system should flag imputed values and exclude them from refund decisions.
- **Mock LLM.** Classification accuracy, retries and agent paths reflect the deterministic mock, not a real model. Few-shot's 100% accuracy is a property of its keyword ordering. A real model needs a held-out labelled set and periodic re-evaluation.
- **Lapse model.** F1 ≈ 0.65 on 232 test rows has wide uncertainty (a few customers move it by several points). Next: time-based validation on several cutoffs, probability calibration and a cost-weighted decision threshold.
- **RAG.** With 15 chunks, TF-IDF is enough, but it misses paraphrases, and fixed 40-word chunks can still split a fact (overlap 0 dropped 2 of 12 answers). Larger policy sets would need sentence-aware chunking, semantic embeddings and answer-level fact checking. The "I could not find" fallback relies on the model; a production system should also refuse when retrieval scores are near zero.
- **Agent.** `get_order` does not check that the order belongs to the requesting customer, because the brief's tool signature takes only an order id; production must bind the customer to the request. The ledger and state are local JSON files with no locking, so production would need a transactional store, authentication and an audit trail for human approvals. As the brief specifies, the 7-day request window is assumed to be met rather than checked.