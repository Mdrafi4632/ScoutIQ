# Exploratory Data Analysis (EDA) for the Soccer Recruitment

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

INPUT_FILE = r"C:\Users\Rafi\Documents\Claude\Projects\Capstone Project\soccer_recruitment_dataset.csv"
OUTPUT_DIR = r"C:\Users\Rafi\Downloads\eda_figures"

# Read the dataset
df = pd.read_csv(INPUT_FILE)

# Remove players without a market value
df = df[df["Market_Value_EUR"].notna()].copy()

# # Convert market value from euros to millions
df["Value_M"] = df["Market_Value_EUR"] / 1000000

# Create a log version of market value
df["logV"] = np.log1p(df["Market_Value_EUR"])

# Age to be numeric
df["Age"] = pd.to_numeric(df["Age"], errors="coerce")

# Get the player's main position
df["PosGrp"] = df["Position_FBref"].astype(str).str.split(",").str[0]
df["PosGrp"] = df["PosGrp"].str.upper()

# Keep only the four main positions
valid_positions = ["FW", "MF", "DF", "GK"]
df = df[df["PosGrp"].isin(valid_positions)]

print("Number of players:", len(df))

print("\nMarket Value Summary (Millions)")
print(df["Value_M"].describe())


# 1. Distribution of player market values
plt.figure(figsize=(7, 4))

plt.hist(df["Value_M"], bins=50, color="#1f77b4", edgecolor="white")

plt.xlabel("Market Value (in Millions)")
plt.ylabel("Number of Players")
plt.title("Distribution of Player Market Values")

plt.tight_layout()
plt.show()


# 2. Distribution of Log Transformed Market Values
plt.figure(figsize=(7, 4))
plt.hist(df["logV"], bins=50, color="#2ca02c", edgecolor="white")

plt.xlabel("Market Value")
plt.ylabel("Number of Players")
plt.title("Distribution of Log Transformed Market Values")

plt.tight_layout()
plt.show()


# 3. Market Value vs Age
plt.figure(figsize=(7, 4))

plt.scatter(df["Age"], df["Value_M"], s=10, alpha=0.4, color="#d62728")

plt.xlabel("Age")
plt.ylabel("Market Value (Millions of Euros)")
plt.title("Market Value vs Age")

plt.tight_layout()
plt.show()


# 4. Market Value vs Goals
plt.figure(figsize=(7, 4))

plt.scatter(df["Goals"], df["Value_M"], s=10, alpha=0.4, color="#9467bd")

plt.xlabel("Goals")
plt.ylabel("Market Value (Millions of Euros)")
plt.title("Market Value vs Goals")

plt.tight_layout()
plt.show()


# 5. Average Market Value by Position
position_avg = df.groupby("PosGrp")["Value_M"].mean().sort_values(ascending=False)

plt.figure(figsize=(7, 4))

position_avg.plot(kind="bar", color="#ff7f0e", edgecolor="white")

plt.xlabel("Position")
plt.ylabel("Average Market Value (Millions of Euros)")
plt.title("Average Market Value by Position")

plt.tight_layout()
plt.show()


# 6. Average Market Value by League
league_avg = df.groupby("League")["Value_M"].mean().sort_values(ascending=False)

plt.figure(figsize=(7, 4))

league_avg.plot(kind="bar", color="#17becf", edgecolor="white")

plt.xlabel("League")
plt.ylabel("Average Market Value (Millions of Euros)")
plt.title("Average Market Value by League")
plt.xticks(rotation=30, ha="right")

plt.tight_layout()
plt.show()


# 7. Top Features Correlated with Market Value
numeric_data = df.select_dtypes(include=[np.number])
numeric_data = numeric_data.drop(columns=["Value_M", "logV"], errors="ignore")

correlation = numeric_data.corr()

top_correlations = correlation["Market_Value_EUR"].drop("Market_Value_EUR")
top_correlations = top_correlations.sort_values(ascending=False).head(15)

plt.figure(figsize=(7, 5))

top_correlations.plot(kind="bar", color="#1f77b4", edgecolor="white")

plt.xlabel("Features")
plt.ylabel("Correlation with Market Value")
plt.title("Top 15 Features Correlated with Market Value")

plt.tight_layout()
plt.show()


# 8. Age Curve by Position
plt.figure(figsize=(8, 5))

positions = ["FW", "MF", "DF", "GK"]

for pos in positions:
    position_data = df[df["PosGrp"] == pos]

    age_curve = position_data.groupby(position_data["Age"].round())["Value_M"].mean()
    age_curve = age_curve[(age_curve.index >= 16) & (age_curve.index <= 36)]

    plt.plot(age_curve.index, age_curve.values, marker="o", label=pos)

plt.xlabel("Age")
plt.ylabel("Average Market Value (Millions of Euros)")
plt.title("Market Value by Age for Each Position")

plt.legend(title="Position")
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

print("\nPeak Market Value Age by Position")

for pos in positions:
    position_data = df[df["PosGrp"] == pos]

    age_curve = position_data.groupby(position_data["Age"].round())["Value_M"].mean()
    age_curve = age_curve[(age_curve.index >= 16) & (age_curve.index <= 34)]

    print(f"{pos}: Peaks at age {int(age_curve.idxmax())} ({age_curve.max():.1f} million)")



# 9. Feature Correlation Heatmap
# Select the features to include in the heatmap
features = [
    "Goals",
    "Assists",
    "Shots",
    "Shots_on_Target",
    "Goals_Plus_Assists",
    "Non_Penalty_Goals",
    "Minutes",
    "Matches_Played",
    "Age",
    "Value_M"
]

# Keep only the features that exist in the dataset
features = [feature for feature in features if feature in df.columns]

# Calculate the correlation matrix
correlation_matrix = df[features].corr()

plt.figure(figsize=(7.5, 6.5))

image = plt.imshow(correlation_matrix, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(image, fraction=0.046)

plt.xticks(range(len(features)), features, rotation=45, ha="right", fontsize=8)
plt.yticks(range(len(features)), features, fontsize=8)

# Display the correlation value inside each cell
for i in range(len(features)):
    for j in range(len(features)):
        value = correlation_matrix.iloc[i, j]

        plt.text(
            j,
            i,
            f"{value:.2f}",
            ha="center",
            va="center",
            fontsize=7,
            color="white" if abs(value) > 0.6 else "black"
        )

plt.title("Feature Correlation Heatmap")

plt.tight_layout()
plt.show()



# 10. Per-90 Statistics vs. Season Totals
# Replace zero full 90s with missing values to avoid division by zero
df["nineties"] = df["Full_90s_Played"].replace(0, np.nan)

comparison_results = []

# Compare season totals and per-90 statistics
for stat in ["Goals", "Assists", "Shots"]:

    df[f"{stat}_per90"] = df[stat] / df["nineties"]

    total_correlation = df[stat].corr(df["Value_M"])
    per90_correlation = df[f"{stat}_per90"].corr(df["Value_M"])

    comparison_results.append([stat, total_correlation, per90_correlation])

# Separate the results for plotting
labels = [row[0] for row in comparison_results]
total_values = [row[1] for row in comparison_results]
per90_values = [row[2] for row in comparison_results]

x = np.arange(len(labels))
width = 0.35

# Create the comparison chart
plt.figure(figsize=(7.4, 5))

plt.bar(x - width/2, total_values, width, label="Season Total", color="#1f77b4")
plt.bar(x + width/2, per90_values, width, label="Per 90 Minutes", color="#ff7f0e")

plt.xticks(x, labels)
plt.ylabel("Correlation with Market Value")
plt.title("Season Totals vs. Per-90 Statistics")

plt.legend()
plt.grid(alpha=0.3, axis="y")

plt.tight_layout()
plt.show()

print("\nCorrelation with Market Value")

for stat, total_corr, per90_corr in comparison_results:
    print(f"{stat}: Total = {total_corr:.2f}, Per 90 = {per90_corr:.2f}")



# 11. League Pricing Premium
# Remove rows with missing values
model_data = df.dropna(subset=["Age", "Goals", "Minutes", "logV"]).copy()

# Create the input features
features = np.column_stack([
    model_data["Age"],
    model_data["Goals"],
    model_data["Minutes"],
    model_data["Age"] ** 2
])

# Train a linear regression model
model = LinearRegression()
model.fit(features, model_data["logV"])

# Calculate the residual values
model_data["Residual"] = model_data["logV"] - model.predict(features)

# Calculate the average residual for each league
league_premium = model_data.groupby("League")["Residual"].mean()
league_premium = league_premium.sort_values(ascending=False)

# Create the bar chart
plt.figure(figsize=(7.5, 4.5))

colors = [
    "#2ca02c" if value > 0 else "#d62728"
    for value in league_premium.values
]

league_premium.plot(
    kind="bar",
    color=colors,
    edgecolor="white"
)

plt.axhline(0, color="black", linewidth=0.8)
plt.xlabel("League")
plt.ylabel("Average Residual Log Value")
plt.title("League Pricing Premium")
plt.xticks(rotation=30, ha="right")

plt.tight_layout()
plt.show()

print("\nLeague Pricing Premium")
print(league_premium.round(3))



# 12. Contract Length vs. Market Value
# Convert contract expiry to a date
df["Contract_Expiry"] = pd.to_datetime(df["Contract_Expiry"], errors="coerce")

# Calculate the number of years remaining on each contract
current_date = pd.Timestamp("2026-01-01")
df["Years_Left"] = (df["Contract_Expiry"] - current_date).dt.days / 365.25

# Keep players with between 0 and 6 years remaining
contract_data = df[(df["Years_Left"] >= 0) & (df["Years_Left"] <= 6)]

# Calculate the average market value by years remaining
average_value = contract_data.groupby(contract_data["Years_Left"].round())["Value_M"].mean()

# Create the line chart
plt.figure(figsize=(7, 4.5))

plt.plot(
    average_value.index,
    average_value.values,
    marker="o",
    color="#9467bd"
)

plt.xlabel("Years Remaining on Contract")
plt.ylabel("Average Market Value (Millions of Euros)")
plt.title("Market Value by Remaining Contract Length")
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

print("\nAverage Market Value by Years Remaining on Contract")
print(average_value.round(1))

