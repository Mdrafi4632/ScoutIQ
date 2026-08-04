import numpy as np
import pandas as pd
import unicodedata
import matplotlib.pyplot as plt
# Scale features before calculating similarity
from sklearn.preprocessing import StandardScaler
# Calculate the similarity between players
from sklearn.metrics.pairwise import cosine_similarity
# Reduce the data to two dimensions for visualization
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from difflib import get_close_matches

# Number of similar players to display
N_RESULTS = 10
# INPUT_FILE = r"C:\Users\Rafi\Downloads\CSIC Capstone Project\player_valuation_dataset.csv"
INPUT_FILE = r"C:\Users\Rafi\Downloads\Transfermarkt dataset\CISC Capstone Final Project\player_valuation_dataset.csv"


df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")
df.columns = df.columns.str.strip()
df = df.rename(columns={"Position": "Position_FBref"})


# Special letters that accent-stripping can't handle 
SPECIAL_LETTERS = {
    "ø": "o", "Ø": "O", "ß": "ss", "ı": "i", "İ": "I",
    "ł": "l", "Ł": "L", "æ": "ae", "Æ": "AE", "œ": "oe", "Œ": "OE",
    "þ": "th", "Þ": "Th", "ð": "d", "Ð": "D", "đ": "d", "Đ": "D",
    "ħ": "h", "’": "'", "‘": "'", "–": "-", "—": "-",
}

# Repair Excel's garbled text, then reduce every name to plain English
def clean_name(value):
    if not isinstance(value, str):
        return value
    for encoding in ("cp1252", "latin-1"):
        try:
            value = value.encode(encoding).decode("utf-8")
            break
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
    # 2. Replace special letters
    value = "".join(SPECIAL_LETTERS.get(ch, ch) for ch in value)
    # 3. Remove accents
    value = unicodedata.normalize("NFKD", value)
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    return value


for col in ["Player", "Club", "League", "Club_Full_Name", "Nation", "Citizenship"]:
    if col in df.columns:
        df[col] = df[col].map(clean_name)

# Convert market value to numeric
df["Market_Value_EUR"] = (
    df["Market_Value_EUR"]
    .astype(str)
    .str.replace(",", "", regex=False)
)
df["Market_Value_EUR"] = pd.to_numeric(
    df["Market_Value_EUR"],
    errors="coerce"
)

# Convert age to numeric
df["Age"] = pd.to_numeric(
    df["Age"],
    errors="coerce"
)

# Get each player's primary position
df["Primary_Pos"] = (
    df["Position_FBref"]
    .astype(str)
    .str.split(",")
    .str[0]
    .str.upper()
)
# Keep players with enough playing time
df = df[df["Full_90s_Played"] >= 5].copy().reset_index(drop=True)


# Create statistics for player similarity
nineties = df["Full_90s_Played"].replace(0, np.nan)

# Statistics to convert into per-90 values
per90_stats = [
    "Goals",
    "Assists",
    "Non_Penalty_Goals",
    "Shots",
    "Shots_on_Target",
    "Crosses",
    "Interceptions",
    "Tackles_Won",
    "Fouls_Committed",
    "Fouls_Drawn",
    "Offsides",
]
for stat in per90_stats:
    if stat in df.columns:
        df[f"{stat}_Per90"] = pd.to_numeric(df[stat]) / nineties

# Statistics already stored as rates
rate_stats = [
    "Shots_on_Target_Pct",
    "Goals_Per_Shot",
    "Points_Per_Match",
    "Plus_Minus_Per90",
]

# Features used to compare player styles
style_features = [
    f"{stat}_Per90"
    for stat in per90_stats
    if f"{stat}_Per90" in df.columns
]

# Add statistics that are already rates
style_features.extend(
    stat
    for stat in rate_stats
    if stat in df.columns
)
# Replace missing values with 0
df[style_features] = df[style_features].fillna(0)


# Scale the player statistics
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[style_features])

# Save the scaled data
scaled_df = pd.DataFrame(
    scaled_features,
    columns=style_features,
    index=df.index
)

# Reduce the data to two dimensions for visualization
pca = PCA(n_components=2, random_state=42)
pca_coordinates = pca.fit_transform(scaled_features)

df["PC1"] = pca_coordinates[:, 0]
df["PC2"] = pca_coordinates[:, 1]

K = 5
km = KMeans(
    n_clusters=K,
    random_state=42,
    n_init=10
)
df["Cluster"] = km.fit_predict(scaled_features)


# Find players with a similar playing style
def find_similar(player_name, n=10):
    # Look for an exact player name match
    matches = df[df["Player"].str.lower() == player_name.lower()]
    # If no exact match is found, try a partial match
    if matches.empty:
        matches = df[df["Player"].str.contains(player_name, case=False, na=False)]
    if matches.empty:
        return None, None

    # Get the selected player's information
    target_idx = matches.index[0]
    target = df.loc[target_idx]

    # Compare only players in the same primary position
    position = target["Primary_Pos"]

    candidate_players = df[
        (df["Primary_Pos"] == position) &
        (df.index != target_idx)
    ]

    # Calculate cosine similarity
    similarity_scores = cosine_similarity(
        scaled_df.loc[[target_idx]].values,
        scaled_df.loc[candidate_players.index].values
    )[0]

    # Create the results table
    results = candidate_players[
        ["Player", "Club", "League", "Age", "Market_Value_EUR"]
    ].copy()

    results["Similarity"] = similarity_scores

    # Return the most similar players
    results = results.sort_values(
        "Similarity",
        ascending=False
    ).head(n)

    return target, results


# Turn a 0-1 similarity score into a plain-English rating
def rating_label(score):
    if score >= 0.90:
        return "Excellent Match"
    elif score >= 0.80:
        return "Very Good"
    elif score >= 0.70:
        return "Good"
    elif score >= 0.55:
        return "Fair"
    else:
        return "Weak"


# Find which stats two players are MOST alike in (smallest scaled gap)
def key_similarities(target_idx, other_idx, top=5):
    gap = (scaled_df.loc[target_idx] - scaled_df.loc[other_idx]).abs()
    closest = gap.sort_values().head(top)
    # Make the feature names readable, e.g. "Goals_Per90" -> "Goals per 90"
    labels = [
        feat.replace("_Per90", " per 90").replace("_", " ")
        for feat in closest.index
    ]
    return labels


# Display the player and similar players
def show_results(player_name, n=10):
    # Find similar players
    target, similar_players = find_similar(player_name, n)

    # Player not found
    if target is None:
        print(f"\n'{player_name}' not found.")
        print("Check the spelling or make sure the player has enough minutes played.\n")
        return

    # Get the player's market value
    market_value = target["Market_Value_EUR"]
    if pd.notna(market_value):
        market_value_text = f"€{market_value/1e6:.1f}M"
    else:
        market_value_text = "N/A"

    # Display selected player
    print(
        f"\nTarget: {target['Player']} "
        f"({target['Club']}, {target['Primary_Pos']}), "
        f"Age {int(target['Age'])}, "
        f"Market Value: {market_value_text}"
    )

    results = similar_players.copy()
    # Convert market value to millions
    results["Value (EUR M)"] = (
        results["Market_Value_EUR"] / 1e6
    ).round(1).fillna("N/A")

    # Show similarity as an easy-to-read percentage + word rating
    results["Match"] = (results["Similarity"] * 100).round(1).astype(str) + "%"
    results["Rating"] = results["Similarity"].apply(rating_label)

    # Mark cheaper alternatives
    if pd.notna(market_value):
        results["Cheaper?"] = np.where(
            results["Market_Value_EUR"] < market_value,
            "YES",
            ""
        )
    else:
        results["Cheaper?"] = ""
    print("\nMOST SIMILAR PLAYERS\n")
    display_cols = [
        "Player",
        "Club",
        "League",
        "Age",
        "Value (EUR M)",
        "Match",
        "Rating",
        "Cheaper?"
    ]
    table = results[display_cols].astype(str)
    # Each column is as wide as its header or its widest value
    widths = {
        col: max(len(col), table[col].map(len).max())
        for col in display_cols
    }
    # Center the header, then center every cell underneath it
    print("   ".join(col.center(widths[col]) for col in display_cols))
    for _, row in table.iterrows():
        print("   ".join(row[col].center(widths[col]) for col in display_cols))

    # Explain WHY the closest player is similar (which stats are most alike)
    top_match = similar_players.iloc[0]
    top_idx = similar_players.index[0]
    shared_stats = key_similarities(target.name, top_idx)

    print(
        f"\nWHY {top_match['Player']} is the closest match "
        f"({top_match['Similarity'] * 100:.1f}% - "
        f"{rating_label(top_match['Similarity'])})"
    )
    print("Most alike in these stats:")
    for stat in shared_stats:
        print(f"   {stat:<26} •")
    print()

    # Display charts
    draw_charts(target, similar_players)


# Charts
def draw_charts(target, similar_players):
    # Chart 1: Similarity vs Market Value
    plt.figure(figsize=(9, 6))
    plt.scatter(
        similar_players["Market_Value_EUR"] / 1e6,
        similar_players["Similarity"],
        s=60
    )
    # Add player names
    for _, player in similar_players.iterrows():
        plt.annotate(
            player["Player"],
            (
                player["Market_Value_EUR"] / 1e6,
                player["Similarity"]
            ),
            fontsize=8,
            xytext=(4, 4),
            textcoords="offset points"
        )
    # Draw a vertical line for the selected player's market value
    if pd.notna(target["Market_Value_EUR"]):
        plt.axvline(
            target["Market_Value_EUR"] / 1e6,
            linestyle="--",
            label=f"{target['Player']} Market Value"
        )

        plt.legend()

    plt.xlabel("Market Value (Million Euros)")
    plt.ylabel("Similarity Score")
    plt.title(f"Players Most Similar to {target['Player']}")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


    # Chart 2: Player Style Map (PCA)
    plt.figure(figsize=(9, 7))
    # Plot all players
    plt.scatter(
        df["PC1"],
        df["PC2"],
        s=8,
        alpha=0.2,
        label="All Players"
    )
    # Highlight similar players
    plt.scatter(
        df.loc[similar_players.index, "PC1"],
        df.loc[similar_players.index, "PC2"],
        s=55,
        label="Similar Players"
    )
    # Add player names
    for player_id in similar_players.index:
        plt.annotate(
            df.loc[player_id, "Player"],
            (
                df.loc[player_id, "PC1"],
                df.loc[player_id, "PC2"]
            ),
            fontsize=7,
            xytext=(3, 3),
            textcoords="offset points"
        )

    # Highlight the selected player
    plt.scatter(
        target["PC1"],
        target["PC2"],
        s=180,
        marker="*",
        label=target["Player"],
        zorder=5
    )
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("Player Style Map (PCA)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# Player similarity search
if __name__ == "__main__":

    print("Player Replacement Finder")
    print("Enter the player name to find similar players")
    print("Type 'quit' to exit.\n")

    while True:

        player_name = input("Search player: ").strip()
        if player_name.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        # Check if the player exists
        if player_name not in df["Player"].values:
            # Find the closest matching player name
            suggestion = get_close_matches(
                player_name,
                df["Player"],
                n=1,
                cutoff=0.6
            )
            if suggestion:
                print(f"\nPlayer not found. Did you mean '{suggestion[0]}'?")
                player_name = suggestion[0]
            else:
                print("\nPlayer not found.")
                continue

        show_results(player_name, N_RESULTS)