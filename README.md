# 🛒 Market Basket Analysis — Retail Insights from Transaction Data

Discover which products customers buy together using **association rule mining** on real grocery transaction data.

## What it does
Takes a simple transactions dataset (`Member_number`, `Date`, `itemDescription`) and follows the full data science lifecycle:

1. **Data cleaning** — standardizes columns, parses dates, drops unusable rows
2. **Exploratory analysis** — top purchased items, trends over time (matplotlib / seaborn)
3. **Basket encoding** — groups items by member + date into transaction baskets, builds a binary matrix
4. **Apriori mining** — finds frequent itemsets with `mlxtend`
5. **Association rules** — generates rules ranked by support, confidence, and lift ("customers who buy X also buy Y")
6. **Exports** — saves frequent itemsets and rules to CSV

## Tech
Python · pandas · matplotlib · seaborn · mlxtend

## Run it
```bash
pip install -r requirements.txt
python market_basket_analysis.py
```

Update `DATA_PATH` at the top of the script to point at your CSV. Expected columns: `Member_number`, `Date` (`DD-MM-YYYY`), `itemDescription`.

Tune `MIN_SUPPORT` (default `0.005`) and `MIN_LIFT` (default `1.0`) to control how many itemsets/rules are returned.

## Sample output
- Top 10 most frequently purchased items (bar chart)
- Basket matrix dimensions and preview
- Top 20 frequent itemsets by support
- Top 20 association rules by lift → confidence → support
- `frequent_itemsets.csv` and `association_rules.csv` written next to the dataset
