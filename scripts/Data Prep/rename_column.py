import pandas as pd

# Local file paths
INPUT_FILE = r"C:\Users\Rafi\Downloads\CSIC Capstone Project\Not Finalized Dataset\fbref_transfermarkt_clean.csv"
OUTPUT_FILE = r"C:\Users\Rafi\Downloads\CSIC Capstone Project\player_valuation_dataset.csv"

# Load cleaned dataset
df = pd.read_csv(INPUT_FILE)

# Rename columns for consistency and readability
rename = {
    "Player":"Player",
    "Nation":"Nation",
    "Pos":"Position_FBref",
    "Squad":"Club",
    "Comp":"League",
    "Age":"Age",
    "Born":"Birth_Year",

    "MP":"Matches_Played",
    "Starts":"Starts",
    "Min":"Minutes",
    "90s":"Full_90s_Played",

    "Gls":"Goals",
    "Ast":"Assists",
    "G+A":"Goals_Plus_Assists",
    "G-PK":"Non_Penalty_Goals",
    "PK":"Penalty_Goals",
    "PKatt":"Penalty_Attempts",

    "CrdY":"Yellow_Cards",
    "CrdR":"Red_Cards",

    "GA":"GK_Goals_Against",
    "GA90":"GK_Goals_Against_Per90",
    "SoTA":"GK_Shots_on_Target_Against",
    "Saves":"GK_Saves",
    "Save%":"GK_Save_Pct",
    "W":"GK_Wins",
    "D":"GK_Draws",
    "L":"GK_Losses",
    "CS":"GK_Clean_Sheets",
    "CS%":"GK_Clean_Sheet_Pct",
    "PKA":"GK_Penalties_Allowed",
    "PKsv":"GK_Penalties_Saved",
    "PKm":"GK_Penalties_Missed",

    "Sh":"Shots",
    "SoT":"Shots_on_Target",
    "SoT%":"Shots_on_Target_Pct",
    "Sh/90":"Shots_Per90",
    "SoT/90":"Shots_on_Target_Per90",
    "G/Sh":"Goals_Per_Shot",
    "G/SoT":"Goals_Per_Shot_on_Target",

    "Mn/MP":"Minutes_Per_Match",
    "Min%":"Pct_of_Minutes",
    "Mn/Start":"Minutes_Per_Start",

    "Compl":"Completed_Matches",
    "Subs":"Sub_Appearances",
    "Mn/Sub":"Minutes_Per_Sub",
    "unSub":"Unused_Sub",
    "PPM":"Points_Per_Match",

    "onG":"Team_Goals_On_Pitch",
    "onGA":"Team_Goals_Against_On_Pitch",
    "+/-":"Plus_Minus",
    "+/-90":"Plus_Minus_Per90",

    "On-Off":"On_Off_Impact",
    "2CrdY":"Second_Yellow_Cards",
    "Fls":"Fouls_Committed",
    "Fld":"Fouls_Drawn",
    "Off":"Offsides",
    "Crs":"Crosses",
    "Int":"Interceptions",
    "TklW":"Tackles_Won",
    "OG":"Own_Goals",

    "player_id":"Transfermarkt_ID",
    "position":"Position_TM",
    "sub_position":"Detailed_Position",
    "current_club_name":"Club_Full_Name",
    "country_of_citizenship":"Citizenship",
    "contract_expiration_date":"Contract_Expiry",
    "foot":"Preferred_Foot",
    "height_in_cm":"Height_cm",

    "market_value_in_eur":"Market_Value_EUR",
    "value_millions_eur":"Market_Value_Millions_EUR",
    "highest_market_value_in_eur":"Highest_Market_Value_EUR"
}

# Check for columns without a rename mapping
missing = [col for col in df.columns if col not in rename]
if missing:
    print("Columns left unchanged:", missing)

# Rename columns
df.rename(columns=rename, inplace=True)

# Save final dataset
df.to_csv(OUTPUT_FILE, index=False)

print("Dataset saved successfully.")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print(f"Columns renamed: {len(rename) - len(missing)}")
