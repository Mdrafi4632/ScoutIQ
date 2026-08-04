import pandas as pd
import unicodedata
import re

# Local file paths
INPUT_FBREF = r"C:\Users\Rafi\Downloads\CSIC Capstone Project\FBref dataset 25-26\FBref_players_dataset_25_26.csv"
INPUT_TM = r"C:\Users\Rafi\Downloads\CSIC Capstone Project\Transfermarkt dataset\players.csv"
OUTPUT_FILE = r"C:\Users\Rafi\Downloads\CSIC Capstone Project\Not Finalized Dataset\fbref_transfermarkt_clean.csv"


# Normalize player names for matching
def normalize_name(name):
    if pd.isna(name):
        return ""

    name = unicodedata.normalize("NFKD", str(name))
    name = "".join(c for c in name if not unicodedata.combining(c))
    name = name.lower()
    name = re.sub(r"[^a-z0-9 ]", " ", name)
    name = re.sub(r"\s+", " ", name).strip()

    return name


# Load FBref dataset
fb = pd.read_csv(INPUT_FBREF)

# Remove duplicate columns
fb = fb.loc[:, ~fb.columns.duplicated()]

fb["norm_name"] = fb["Player"].map(normalize_name)
fb["birth_year"] = pd.to_numeric(fb["Born"], errors="coerce")

print("FBref players:", len(fb))


# Load Transfermarkt dataset
tm = pd.read_csv(INPUT_TM)

tm["norm_name"] = tm["name"].map(normalize_name)
tm["birth_year"] = pd.to_datetime(
    tm["date_of_birth"],
    errors="coerce"
).dt.year

# Keep the latest / most valuable record for each player
tmv = tm.dropna(subset=["market_value_in_eur"]).copy()
tmv = tmv.sort_values(
    ["last_season", "market_value_in_eur"],
    ascending=False
)

tm_key = tmv.drop_duplicates(
    subset=["norm_name", "birth_year"],
    keep="first"
)

tm_cols = [
    "player_id",
    "name",
    "norm_name",
    "birth_year",
    "market_value_in_eur",
    "highest_market_value_in_eur",
    "position",
    "sub_position",
    "current_club_name",
    "country_of_citizenship",
    "contract_expiration_date",
    "foot",
    "height_in_cm"
]


# Match players using normalized name and birth year
merged = fb.merge(
    tm_key[tm_cols],
    on=["norm_name", "birth_year"],
    how="left",
    suffixes=("_fbref", "_tm")
)

matched = merged["market_value_in_eur"].notna()

print(
    f"Primary matches: {matched.sum()} / {len(merged)} "
    f"({matched.mean()*100:.1f}%)"
)


# Match remaining players using unique names
unmatched = merged[~matched].copy()

name_counts = tm_key["norm_name"].value_counts()
unique_names = set(name_counts[name_counts == 1].index)

tm_nameonly = (
    tm_key[tm_key["norm_name"].isin(unique_names)]
    .set_index("norm_name")
)

filled_idx = []

for i, row in unmatched.iterrows():

    player_name = row["norm_name"]

    if player_name in tm_nameonly.index:

        for col in tm_cols:

            if col in ("norm_name", "birth_year"):
                continue

            merged.at[i, col] = tm_nameonly.at[player_name, col]

        filled_idx.append(i)

print(f"Fallback matches: {len(filled_idx)}")

final_matched = merged["market_value_in_eur"].notna()

print(
    f"Total matched: {final_matched.sum()} / {len(merged)} "
    f"({final_matched.mean()*100:.1f}%)"
)


# Convert market value to millions
merged["value_millions_eur"] = (
    merged["market_value_in_eur"] / 1e6
)


# Save merged dataset
merged.to_csv(OUTPUT_FILE, index=False)

print("Merged dataset saved successfully.")
print("Final dataset shape:", merged.shape)


# Quick validation
cols = [
    "Player",
    "Squad",
    "Comp",
    "birth_year",
    "Gls",
    "Ast",
    "value_millions_eur",
    "current_club_name"
]

print("\nSample matched players:")
print(merged[final_matched][cols].head(10).to_string(index=False))

print("\nSample unmatched players:")
print(
    merged[~final_matched][["Player", "Squad", "birth_year"]]
    .head(10)
    .to_string(index=False)
)
