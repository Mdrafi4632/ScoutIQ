import pandas as pd

INPUT_FILE = r"C:\Users\Rafi\Downloads\fbref_transfermarkt_merged.csv"
OUTPUT_FILE  = r"C:\Users\Rafi\Downloads\fbref_transfermarkt_clean.csv"

# Load FBref + Transfermarkt dataset
df = pd.read_csv(INPUT_FILE)
print("Before:", df.shape)

# Remove duplicate metadata and helper columns
drop_cols = ["PK", "norm_name", "name", "birth_year", "GA-PK"]
df = df[[c for c in df.columns if "_stats_" not in c and c not in drop_cols]]

# Organize columns into a cleaner order
identity = ["Player", "Nation", "Pos", "Squad", "Comp", "Age", "Born"]

tm_cols = [
    "player_id", "position", "sub_position", "current_club_name",
    "country_of_citizenship", "contract_expiration_date",
    "foot", "height_in_cm"
]

target = [
    "market_value_in_eur",
    "value_millions_eur",
    "highest_market_value_in_eur"
]

# Keep all remaining feature columns in the middle
other_cols = [c for c in df.columns if c not in identity + tm_cols + target]
df = df[identity + other_cols + tm_cols + target]

print("After:", df.shape)

# Save cleaned dataset
df.to_csv(OUTPUT_FILE, index=False)

print("Saved fbref_transfermarkt_clean.csv")
print("Players with market value:", df["market_value_in_eur"].notna().sum())
