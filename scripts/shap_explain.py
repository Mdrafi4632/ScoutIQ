# Model Explainability with SHAP
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import xgboost as xgb
from xgboost import XGBRegressor

# dataset
INPUT_FILE = r"C:\Users\Rafi\Downloads\Transfermarkt dataset\CISC Capstone Final Project\player_valuation_dataset.csv"


# Load data
df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")
df.columns = df.columns.str.strip()
df = df.rename(columns={"Position": "Position_FBref"})

df["Market_Value_EUR"] = (
    df["Market_Value_EUR"]
    .astype(str)
    .str.replace(",", "", regex=False)
)

df["Market_Value_EUR"] = pd.to_numeric(df["Market_Value_EUR"], errors="coerce")
df = df[df["Market_Value_EUR"].notna()].copy()
df["Age"] = pd.to_numeric(df["Age"], errors="coerce")



# Feature Engineering
df["Primary_Pos"] = (
    df["Position_FBref"]
    .astype(str)
    .str.split(",")
    .str[0]
    .str.upper()
)

# Calculate years remaining on the player's contract
expiry_date = pd.to_datetime(
    df["Contract_Expiry"],
    errors="coerce"
)
df["Years_Left"] = (
    expiry_date - pd.Timestamp("2026-01-01")
).dt.days / 365.25

# Squared age feature
df["Age2"] = df["Age"] ** 2


# Target variable
y = np.log1p(df["Market_Value_EUR"])

# Columns that are NOT features
drop_cols = [
    "Player", "Nation", "Position_FBref", "Club", "Club_Full_Name",
    "Position_TM", "Detailed_Position", "Citizenship", "Contract_Expiry",
    "Preferred_Foot", "Transfermarkt_ID", "Birth_Year", "Market_Value_EUR",
    "Market_Value_Millions_EUR", "Highest_Market_Value_EUR",
]

num_features = df.drop(
    columns=drop_cols,
    errors="ignore"
).select_dtypes(include=[np.number])

# Convert categorical features into dummy variables
cat_features = pd.get_dummies(
    df[["League", "Primary_Pos"]],
    drop_first=True
)

# Combine all features
X = pd.concat(
    [num_features, cat_features],
    axis=1
).fillna(0)

print("Feature Matrix Shape:", X.shape)


# Train Model
model = XGBRegressor(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42
)
model.fit(X, y)


# Compute SHAP values
print("Calculating SHAP values: ")
booster = model.get_booster()
dmatrix = xgb.DMatrix(X, feature_names=list(X.columns))
contribs = booster.predict(dmatrix, pred_contribs=True)
base_values = contribs[:, -1]
shap_matrix = contribs[:, :-1]

# Create a SHAP explanation object
shap_values = shap.Explanation(
    values=shap_matrix,
    base_values=base_values,
    data=X.values,
    feature_names=list(X.columns)
)


# Chart 1: SHAP Summary Plot
plt.figure()
shap.summary_plot(shap_values, X, show=False, max_display=15)
plt.title("SHAP Summary Plot")
plt.tight_layout()
plt.show()

# CHART 2: SHAP Feature Importance
plt.figure()
shap.summary_plot(shap_values, X, plot_type="bar", show=False, max_display=15)
plt.title("Feature Importance (SHAP)")
plt.tight_layout()
plt.show()


# Per player explanation
def explain_player(player_name, top=8):
    # Find the player
    matches = df[df["Player"].str.contains(player_name, case=False, na=False)]
    if matches.empty:
        print(f"\n'{player_name}' not found.")
        return

    # Get the selected player
    row_pos = df.index.get_loc(matches.index[0])
    player = df.loc[matches.index[0]]

    # Predict market value
    predicted_value = np.expm1(
        model.predict(X.iloc[[row_pos]])[0]
    )

    actual_value = player["Market_Value_EUR"]

    print(f"\nPredicted Market Value: €{predicted_value/1e6:.1f}M")

    print(f"Actual Market Value: €{actual_value/1e6:.1f}M")

    # Check if the player is undervalued or overvalued
    difference = predicted_value - actual_value
    if difference > 0:
        print(f"\nVerdict: UNDERVALUED by €{difference/1e6:.1f}M")
    else:
        print(f"\nVerdict: OVERVALUED by €{abs(difference)/1e6:.1f}M")

    # SHAP values for this player
    contributions = pd.Series(shap_values.values[row_pos],index=X.columns)

    # Sort features from most important to least important
    contributions = contributions.reindex(contributions.abs().sort_values(ascending=False).index)


    print("\nTop Factors Affecting the Prediction\n")
    # Build a centered table
    rows = [("Feature", "Player value", "Effect", "SHAP Value")]
    for feature in contributions.head(top).index:
        effect = "Increases" if contributions[feature] > 0 else "Decreases"
        rows.append((
            feature,
            f"{X.iloc[row_pos][feature]:.2f}",
            effect,
            f"{contributions[feature]:.2f}",
        ))
    widths = [max(len(r[i]) for r in rows) for i in range(4)]
    for r in rows:
        print("   ".join(cell.center(widths[i]) for i, cell in enumerate(r)))
    print()
    plt.figure()
    shap.plots.waterfall(
        shap_values[row_pos],
        max_display=12,
        show=False
    )
    plt.title(f"SHAP Breakdown: {player['Player']}")
    plt.tight_layout()
    plt.show()


# SHAP Player Explainer
if __name__ == "__main__":

    print("SHAP Player Explainer")
    print("Enter a player's name to explain their predicted market value.")
    print("Type 'quit' or 'q' to exit.\n")

    while True:
        player_name = input("Search player: ").strip()
        if player_name.lower() in ["quit", "q"]:
            print("Goodbye!")
            break
        explain_player(player_name)

