# Instamart Customer Intelligence & Support Copilot

## Overview

An internal Copilot for the (fictional) Swiggy Instamart customer-experience team, built in four connected Parts on fully synthetic data. All amounts are in Indian Rupees (INR).

| Part | What it does | Main script(s) | Feeds |
|---|---|---|---|
| A | Audits the dirty raw exports, cleans them with documented rules, stores them as CSV + SQLite, answers 7 SQL questions and cross-checks them in pandas | `src/part_a_clean.py`, `src/part_a_validate.py`, `src/part_a_analysis.py`, `sql/queries.sql` | B (clean CSVs), D (`data/instamart.db`) |
| B | RFM segmentation with K-means (K=4, named segments, Premium flag) and a leakage-free lapse model | `src/part_b_segments.py`, `src/part_b_lapse.py` | C (`outputs/customer_segments.csv`) |
| C | Support-ticket triage: input guardrails, PII redaction, zero/few-shot prompts, retries, output validation, token budget, segment-aware routing | `src/part_c_triage.py` | — |
| D | Policy RAG assistant (TF-IDF retrieval + grounded answers) and a refund agent with tools over `data/instamart.db`, guardrails, step/token limits, an idempotent ledger and crash-resume | `src/policy_rag.py`, `src/part_d_agent.py` | — |

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
11. `python src/policy_rag.py` → policy QA evaluation on `data/raw/policy_eval.csv`
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

Every call goes through `call_with_retry`: at most 3 retries, wait before retry n = `0.05 × 2**n × uniform(0.9, 1.15)` s, with one log line per retry, for example `[retry] ticket=T0002 retry=0 wait=0.053s error=503 server busy`. `mock_llm.reset()` is called before every run. Each full run hit **19 TransientLLMErrors, all recovered on the first retry; 0 permanent failures.**

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

All 21 zero-shot errors are fraud or technical tickets mislabelled as billing, because they also mention "charged" or "payment". Eleven of those are fraud cases that would miss the 2-hour security SLA. Few-shot fixes all 21 but costs 2.2× the tokens (+15,912 in total, about 108 per ticket). That is worth paying here: one misrouted fraud ticket is far more expensive than 108 tokens. Zero-shot would only be enough for a cheap first pass whose billing labels are always re-checked.

### Routing (`outputs/triage_results.csv`, few-shot run)

Each ticket is joined to `outputs/customer_segments.csv` on customer_id. Routing follows `escalation_policy.txt`:

| Rule | Queue | Priority | SLA |
|---|---|---|---|
| fraud | security_team | high | 2 h (1 h if premium) |
| billing / delivery / technical | billing_desk / delivery_ops / tech_support | high if premium else normal | 1 h premium, 24 h others |
| flagged_invalid_output, flagged_long, skipped_empty | human_review | high if premium else normal | 1 h premium, 24 h others |

Result: security_team 34 (3 premium); billing_desk 38; delivery_ops 36; tech_support 39; human_review 17. In total 16 tickets come from Premium customers and get the 1-hour SLA. Columns: ticket_id, prompt, customer_id, true_category, status, category, retries, tokens, segment_name, is_premium, queue, priority, sla_hours, redacted_text. Only redacted text is stored.

### Token budget demo (`outputs/triage_budget_demo.csv`)

The same few-shot pipeline runs with `TOKEN_BUDGET = 3000`. Before each call, the projected prompt tokens are checked against what remains; if they don't fit, the ticket gets `skipped_budget` and is never sent. Result: 15 tickets classified (14 json + 1 keyword) using 2,843 tokens, and 141 tickets `skipped_budget`, the first at file row 18 (T0123). The 8 empty/long tickets are still caught by the input guardrails first. Spend never exceeds the budget. Columns include budget, tokens and cumulative_tokens per ticket.

## Part D

### Policy RAG assistant (`src/policy_rag.py`)

- **Chunks:** the 5 policy files are split into 34 sentences, each tagged with its source file.
- **Retrieval:** TF-IDF (English stop words, unigrams + bigrams, sublinear tf) with cosine similarity; top-3 chunks are sent as `CONTEXT: [file.txt] sentence … QUESTION: …` to the `POLICY ASSISTANT`. The answer cites the file of the sentence it used.
- **Refusal guard:** if the best chunk scores below 0.10, the assistant answers "I could not find that information in the policy documents." without calling the model.

| Metric on `policy_eval.csv` | Result |
|---|---|
| retrieval hit@3 (expected file among top 3) | 12 / 12 |
| top-1 file correct | 12 / 12 |
| answer contains expected fact | 12 / 12 |
| out-of-scope questions refused (mascot colour, cricket final) | 2 / 2 (best score 0.000) |
| total tokens for 14 questions | 1,615 |

### Refund agent (`src/part_d_agent.py`)

ReAct loop over the `AGENT` mock with five tools:

| Tool | What it does |
|---|---|
| `get_order` | Read-only SQL on `data/instamart.db`; returns not-found when the order belongs to another customer |
| `search_policy` | Reuses the RAG retriever above |
| `calculate_refund` | Full amount if delivery took more than 45 min, else 0 |
| `issue_refund` | Appends to `outputs/refund_ledger.jsonl` |
| `escalate_to_human` | Records the escalation in the ledger |

Guardrails enforced in code, whatever the model says:

- `issue_refund` and `escalate_to_human` re-derive the allowed refund from the database and refuse any mismatch, any non-completed order and any order not verified for this customer.
- `issue_refund` refuses anything above INR 1000.
- The ledger is idempotent: an order is never paid twice.
- Runs are capped at 6 model steps (per `escalation_policy.txt`) and 4,000 tokens; a run that hits either cap is handed to a human with a summary.
- An infrastructure failure in a tool stops the run.
- After every tool execution the full run state is checkpointed atomically to `state/<run_id>.json`, and every run is logged to `outputs/agent_runs.jsonl`.

Seven scenarios (`data/raw/agent_scenarios.csv`):

| Scenario | Customer | Expected | Outcome | Refund (INR) | Steps | Tokens | Tools executed |
|---|---|---|---|---|---|---|---|
| S1 | C0007 | REFUND 128.25 | REFUND | 128.25 | 5 | 708 | get_order > search_policy > calculate_refund > issue_refund |
| S2 | C0008 | REFUND 135.00 | REFUND | 135.00 | 5 | 708 | get_order > search_policy > calculate_refund > issue_refund |
| S3 | C0016 | ESCALATE 1999.00 | ESCALATE | 1,999.00 (to human) | 5 | 695 | get_order > search_policy > calculate_refund > escalate_to_human |
| S4 | C0001 | DENIED | DENIED (on time) | 0.00 | 4 | 497 | get_order > search_policy > calculate_refund |
| S5 | C0003 | DENIED | DENIED (cancelled) | 0.00 | 3 | 313 | get_order > search_policy |
| S6 | C0001 | NOT_FOUND | NOT_FOUND | 0.00 | 2 | 144 | get_order |
| S7 | C0007 | FALLBACK | FALLBACK (6-step cap) | 0.00 | 6 | 864 | search_policy × 6 |

All 7 match. Ledger after all runs: 3 refunds issued (S1, S2 and the crash demo) totalling INR 623.25, plus 1 escalation (S3, INR 1,999.00).

### Stop-safely demonstrations

| Demo | Setup | Status | Detail |
|---|---|---|---|
| Token budget | S1 with a 300-token budget | **STOPPED_BUDGET** | stopped after 3 steps / 330 tokens (get_order, search_policy, calculate_refund); no refund issued; handed to a human |
| Tool failure | S1 with the database unavailable | **STOPPED_TOOL_ERROR** | get_order raised `OperationalError: unable to open database file`; stopped after 1 step instead of reporting a false "not found" |

An extra idempotency check repeats S1 after it was paid: status `ALREADY_REFUNDED`, and the ledger still holds exactly 1 refund for ORD-100048.

### Crash and resume

A fresh eligible order (ORD-100072, customer C0008, INR 360.00, delivered late) is processed with a crash injected after the 2nd tool execution. The resumed run reloads `state/DEMO_CRASH.json` and continues from the saved transcript without repeating any tool.

- tools executed before the crash: `get_order`, `search_policy`
- tools executed after the resume: `calculate_refund`, `issue_refund`
- final status: REFUND INR 360.00, with exactly 1 ledger entry for ORD-100072

## Limitations and Next Steps

- **The retention list is illustrative.** The Premium flag is a post-hoc whole-window label: it comes from RFM over April–September, including the September window that defines the lapse label. The retention list is therefore illustrative, not a leakage-free evaluation. Next step: recompute segments on pre-cutoff orders only.
- **Synthetic data.** Every pattern (e.g. three customer archetypes, delivery ~N(24, 8)) comes from `generate_data.py`, so model scores and segment shapes will not transfer to real Instamart data.
- **Imputation choices.** City-median ages and category-median delivery times flatten variance. 774 completed orders carry an imputed time of about 25 minutes, so a missing time is assumed on time and can never trigger a late-delivery refund. A production system should flag imputed values and exclude them from refund decisions.
- **Mock LLM.** Classification accuracy, retries and agent paths reflect the deterministic mock, not a real model. Few-shot's 100% accuracy is a property of its keyword ordering. A real model needs a held-out labelled set and periodic re-evaluation.
- **Lapse model.** F1 ≈ 0.65 on 232 test rows has wide uncertainty (a few customers move it by several points). Next: time-based validation on several cutoffs, probability calibration and a cost-weighted decision threshold.
- **RAG.** The 34-sentence corpus makes TF-IDF sufficient. Larger, overlapping policies would need better chunking, a tuned refusal threshold and answer-level fact checking.
- **Agent.** The ledger and state are local JSON files with no locking; production would need a transactional store, authentication and an audit trail for human approvals. The refund-request window (7 days) is not enforced because the data has no delivery dates relative to "today".
