"""
Market Basket Analysis — Grocery Dataset
----------------------------------------
This beginner-friendly project identifies items that are frequently purchased
together using the Apriori algorithm and association rules.

Before running:
1. Install the required libraries:
   pip install pandas matplotlib mlxtend
2. Update DATA_PATH below if your CSV file is stored somewhere else.
3. Run this file with Python.

Expected CSV columns:
    Member_number, Date, itemDescription
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules


# Change this path if your dataset is stored in a different location.
DATA_PATH = Path(r"D:/Project/Groceries_dataset.csv")

# Apriori / association-rule settings
MIN_SUPPORT = 0.005
MIN_LIFT = 1.0


def main():
    # 1. Load the dataset
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {DATA_PATH}\n"
            "Update DATA_PATH near the top of this script to your CSV location."
        )

    df = pd.read_csv(DATA_PATH)

    # 2. Standardize column names for easier use
    required_columns = {"Member_number", "Date", "itemDescription"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Missing expected column(s): {sorted(missing_columns)}. "
            f"Columns found: {list(df.columns)}"
        )

    df = df.rename(
        columns={
            "Member_number": "Member",
            "itemDescription": "Item",
        }
    )

    # Parse dates. The format matches dates such as 01-12-2015.
    df["Date"] = pd.to_datetime(
        df["Date"], format="%d-%m-%Y", errors="coerce"
    )

    # Remove rows that cannot be used in the analysis.
    df = df.dropna(subset=["Member", "Date", "Item"]).copy()
    df["Item"] = df["Item"].astype(str).str.strip()
    df = df[df["Item"] != ""]

    # 3. Basic exploratory data analysis (EDA)
    print("\n========== DATA OVERVIEW ==========")
    print(f"Dataset shape (rows, columns): {df.shape}")
    print("\nData types:")
    print(df.dtypes)
    print("\nMissing values:")
    print(df.isna().sum())
    print(f"\nUnique members: {df['Member'].nunique()}")
    print(f"Unique items: {df['Item'].nunique()}")
    print(f"Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")

    top_items = df["Item"].value_counts().head(10)
    print("\nTop 10 most frequently recorded items:")
    print(top_items)

    # Optional chart: most frequently recorded items
    plt.figure(figsize=(10, 6))
    top_items.sort_values().plot(kind="barh")
    plt.title("Top 10 Most Frequently Purchased Items")
    plt.xlabel("Number of records")
    plt.ylabel("Item")
    plt.tight_layout()
    plt.show()

    # 4. Create a transaction ID from each member and purchase date.
    # This treats all items bought by the same member on the same date as one basket.
    df["BasketID"] = (
        df["Member"].astype(str)
        + "_"
        + df["Date"].dt.strftime("%Y-%m-%d")
    )

    # 5. Convert transactions into a one-hot basket matrix:
    # rows = baskets, columns = items, values = 1 if an item is in the basket.
    basket = (
        df.groupby(["BasketID", "Item"])["Item"]
        .count()
        .unstack(fill_value=0)
    )
    basket = basket.gt(0).astype(int)

    print("\n========== BASKET MATRIX ==========")
    print(f"Number of baskets: {basket.shape[0]}")
    print(f"Number of distinct items: {basket.shape[1]}")
    print("\nBasket matrix preview:")
    print(basket.head())

    # 6. Find frequent itemsets using Apriori.
    frequent_itemsets = apriori(
        basket.astype(bool),
        min_support=MIN_SUPPORT,
        use_colnames=True,
    )

    if frequent_itemsets.empty:
        print(
            "\nNo frequent itemsets found. Try lowering MIN_SUPPORT "
            "(for example, to 0.003)."
        )
        return

    frequent_itemsets["itemsets"] = frequent_itemsets["itemsets"].apply(
        lambda items: ", ".join(sorted(items))
    )
    frequent_itemsets = frequent_itemsets.sort_values(
        "support", ascending=False
    )

    print("\n========== TOP FREQUENT ITEMSETS ==========")
    print(frequent_itemsets.head(20).to_string(index=False))

    # 7. Generate association rules using lift.
    # Use the original itemsets DataFrame (with frozensets) for rule generation.
    itemsets_for_rules = apriori(
        basket.astype(bool),
        min_support=MIN_SUPPORT,
        use_colnames=True,
    )
    rules = association_rules(
        itemsets_for_rules,
        metric="lift",
        min_threshold=MIN_LIFT,
    )

    if rules.empty:
        print("\nNo association rules found with the current settings.")
        return

    # Make the rules easier to read.
    rules["antecedents"] = rules["antecedents"].apply(
        lambda items: ", ".join(sorted(items))
    )
    rules["consequents"] = rules["consequents"].apply(
        lambda items: ", ".join(sorted(items))
    )
    rules = rules.sort_values(
        ["lift", "confidence", "support"], ascending=False
    )

    columns_to_show = [
        "antecedents",
        "consequents",
        "support",
        "confidence",
        "lift",
    ]

    print("\n========== TOP ASSOCIATION RULES ==========")
    print(rules[columns_to_show].head(20).to_string(index=False))

    # 8. Save results in CSV files next to the dataset.
    output_folder = DATA_PATH.parent
    frequent_itemsets.to_csv(
        output_folder / "frequent_itemsets.csv", index=False
    )
    rules[columns_to_show].to_csv(
        output_folder / "association_rules.csv", index=False
    )

    print("\nAnalysis complete.")
    print(f"Frequent itemsets saved to: {output_folder / 'frequent_itemsets.csv'}")
    print(f"Association rules saved to: {output_folder / 'association_rules.csv'}")


if __name__ == "__main__":
    main()
