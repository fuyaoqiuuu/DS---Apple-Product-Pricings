# Apple Product Price Retention Modeling — Executive Summary

**Dataset:** `apple_products_pricing_2020_2026.csv` (80,000 listings, 31 models, 2020–2026, Amazon/Flipkart)
**EDA:** `Apple_Pricing_EDA.ipynb`

## 1. Why this business problem

The original goal, predicting `Current_Price_USD`, is dominated by product scale: launch price alone correlates ~0.95 with current price, and `Model_Name` explains 91% of price variance. A model would mostly re-learn "expensive products cost more" — accurate but uninformative.

Reframing the target as `Retention_Ratio = Current_Price_USD / Launch_Price_USD` removes the scale component and forces the model to learn what the business actually needs: **depreciation behavior** — how much value a product retains N months after launch, by category, condition, and season. This directly serves trade-in, leasing, insurance, and refurbishing use cases, and nothing is lost: predicted price = ratio × known launch price. The ratio is also a better-behaved regression target (near-symmetric vs. skew 1.11 for raw price).

**Hypothesis:** Apple products depreciate along category-specific exponential curves; condition applies a stable multiplier; festival events cause temporary, recoverable dips.

## 2. Deployability and cardinality

`Product_Category` is too coarse — the data has 9 launch price points across 31 models (an $799 iPhone 15 and a $1,199 iPhone 13 Pro Max are both "iPhone"). `Model_Name` is precise but undeployable: a one-hot or target-encoded model name has no representation for a product that didn't exist at training time, and the model must score future launches like *iPhone 19 128GB*.

**Resolution:** decompose the name into generalizable attributes — `Product_Category` + `variant_tier` (Pro / Pro Max / Air / …) + `storage_gb` + `generation`. Every component of a future model was seen in training, so an unseen name maps cleanly to known feature values. The notebook quantifies (via eta²) how much of `Model_Name`'s explanatory power this decomposition recovers.

## 3. Why `Reviews_Count` is dropped

Because the dataset is synthetic, properties a real marketplace guarantees were tested rather than assumed. If reviews were real, the same model's count would grow monotonically over time. It doesn't: counts jump randomly between consecutive listings (e.g., 40 → 84 → 110 → 35 within days), per-model correlation with time is ~0, and the column even correlates *negatively* with price — a generator artifact, not a popularity signal. A column that is neither a time proxy nor a popularity proxy is noise; keeping it risks the model fitting spurious patterns that no future data will reproduce.

Also dropped at load for leakage/redundancy: `Current_Price_INR` (target in another currency), `Discount_Pct` (arithmetically derived from the target), `Launch_Price_INR` (collinear with `Launch_Price_USD`).

## 4. Train/test split and group leakage

Each model appears ~2,500 times across dates and platforms. A random row split scatters the same model — often the same model in the same week — across train and test, so the model can memorize model-specific price levels and validation scores overstate production performance.

Split by **time** instead: train on earlier dates, test on later ones. This mirrors deployment (predicting future dates), and models launched late in the data (e.g., iPhone 17) become natural unseen-model test cases. As a complementary check, run **grouped CV** holding out entire `Model_Name` groups to measure unseen-model generalization specifically.

Report both numbers — time-split for expected production accuracy, group-split for new-launch robustness.
