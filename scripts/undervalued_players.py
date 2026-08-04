import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_predict, KFold
from xgboost import XGBRegressor

INPUT_FILE = r"C:\Users\Rafi\Downloads\Transfermarkt dataset\CISC Capstone Final Project\player_valuation_dataset.csv"
OUTPUT_FILE = r"C:\Users\Rafi\Downloads\Transfermarkt dataset\CISC Capstone Final Project\top_undervalued_players.csv"

# Load the dataset
df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")
df.columns = df.columns.str.strip()
df = df.rename(columns={"Position": "Position_FBref"})
df = df[df["Market_Value_EUR"].notna()].copy()


# Feature Engineering
df["Primary_Pos"] = (
    df["Position_FBref"]
    .astype(str)
    .str.split(",")
    .str[0]
    .str.upper()
)
# Calculate years remaining on each player's contract
expiry_date = pd.to_datetime(
    df["Contract_Expiry"],
    format="%d/%m/%Y"
)
df["Years_Left"] = (expiry_date - pd.Timestamp("2026-01-01")).dt.days / 365.25
# Create a squared age feature
df["Age2"] = df["Age"] ** 2



# Convert Market_Value_EUR to numeric
df["Market_Value_EUR"] = (
    df["Market_Value_EUR"]
    .astype(str)
    .str.replace(",", "", regex=False)
)
df["Market_Value_EUR"] = pd.to_numeric(df["Market_Value_EUR"])

# Create the target variable
y = np.log1p(df["Market_Value_EUR"])
# Columns that won't be used for training
drop_cols = [
    "Player", "Nation", "Position_FBref", "Club", "Club_Full_Name",
    "Position_TM", "Detailed_Position", "Citizenship", "Contract_Expiry", "Preferred_Foot", "Transfermarkt_ID",
    "Birth_Year", "Market_Value_EUR", "Market_Value_Millions_EUR", "Highest_Market_Value_EUR"
]
num_features = df.drop(
    columns=[col for col in drop_cols if col in df.columns]
).select_dtypes(include=[np.number])

# Convert categorical features into dummy variables
cat_features = pd.get_dummies(
    df[["League", "Primary_Pos"]],
    drop_first=True
)
# Combine all features into one dataset
X = pd.concat([num_features, cat_features], axis=1).fillna(0)



# Create 5-fold cross-validation
cv = KFold(n_splits=5, shuffle=True, random_state=42)
# Train the XGBoost model
model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.03,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.7,
    random_state=42
)
# Predict each player's market value using cross-validation
pred_log = cross_val_predict(model, X, y, cv=cv)
# Convert the predicted values back to Euros
df["Predicted_Value"] = np.expm1(pred_log)
# Calculate the difference between predicted and actual market values
df["Gap"] = df["Predicted_Value"] - df["Market_Value_EUR"]
df["Value_Ratio"] = df["Predicted_Value"] / df["Market_Value_EUR"]
# Keep players with meaningful playing time
eligible_players = df[df["Full_90s_Played"] >= 5].copy()



# Rank the most undervalued players
undervalued = eligible_players.sort_values("Gap", ascending=False)
cols = [
    "Player",
    "Primary_Pos",
    "Club",
    "League",
    "Age",
    "Market_Value_EUR",
    "Predicted_Value",
    "Gap",
    "Value_Ratio"
]
# Get the top 20 undervalued players
top20 = undervalued[cols].head(20).copy()
for col in ["Market_Value_EUR", "Predicted_Value", "Gap"]:
    top20[col] = (top20[col] / 1e6).round(1)
# Round the value ratio
top20["Value_Ratio"] = top20["Value_Ratio"].round(2)

print("Top 20 Undervalued Players\n")
print(top20.to_string(index=False))

# Save the full ranked list
undervalued[cols].to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
print(f"\nSaved full ranked list to: {OUTPUT_FILE}")




# Create a bar chart of the top 15 undervalued players
top15 = top20.head(20).iloc[::-1]
plt.figure(figsize=(8, 10))
plt.barh(
    top15["Player"],
    top15["Gap"],
    color="#1e9bbb",
    edgecolor="white"
)
plt.xlabel("Undervaluation Gap in Millions")
plt.ylabel("Player")
plt.title("Top 20 Undervalued Players")
plt.tight_layout()
plt.show()





# Validation Against Real Transfer Fees
import unicodedata
import re

# Load the transfer dataset
TRANSFERS_FILE = r"C:\Users\Rafi\Downloads\Transfermarkt dataset\CISC Capstone Final Project\transfers.csv"

# Standardize player names for matching
def normalize_name(name):
    if pd.isna(name):
        return ""
    name = unicodedata.normalize("NFKD", str(name))
    name = "".join(char for char in name if not unicodedata.combining(char))
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", "", name.lower())).strip()

transfers = pd.read_csv(TRANSFERS_FILE, encoding="utf-8-sig")

# Keep only valid transfer records
transfers = transfers[
    (transfers["transfer_fee"].notna()) &
    (transfers["transfer_fee"] > 0) &
    (transfers["market_value_in_eur"] > 0)
].copy()

# Create a standardized player name for matching
transfers["Player_Key"] = transfers["player_name"].map(normalize_name)
# Calculate the transfer fee relative to the market value
transfers["Premium"] = (
    transfers["transfer_fee"] /
    transfers["market_value_in_eur"]
)

# Calculate the baseline rate of premium transfers
baseline_rate = (transfers["Premium"] > 1).mean()
# Create standardized player names for matching
eligible_players["Player_Key"] = eligible_players["Player"].map(normalize_name)
# Get the top 100 undervalued players
top100 = eligible_players.sort_values("Gap", ascending=False).head(100)
# Match the top 100 players with the transfer dataset
matched_players = transfers[
    transfers["Player_Key"].isin(set(top100["Player_Key"]))
]
# Calculate the percentage of matched players sold above market value
hit_rate = (matched_players["Premium"] > 1).mean()

print("\nValidation Against Real Transfer Fees")
print(
    f"Baseline - All transfers sold above market value: "
    f"{baseline_rate * 100:.1f}%"
)
print(
    f"Top 100 undervalued players found in transfer records: "
    f"{matched_players['Player_Key'].nunique()}"
)
print(
    f"Of those, sold above market value: "
    f"{hit_rate * 100:.1f}% "
    f"(Median Premium: {matched_players['Premium'].median():.2f}x)"
)

# Compare the baseline with the undervalued players
plt.figure(figsize=(6, 4))
plt.bar(
    ["All Transfers\n(Baseline)", "Undervalued\nPlayers"],
    [baseline_rate * 100, hit_rate * 100],
    color=["#999999", "#16bea2"],
    edgecolor="white"
)
plt.ylabel("Players Sold Above Market Value (%)")
plt.title("Transfer Validation Results")
plt.ylim(0, 100)
plt.tight_layout()
plt.show()