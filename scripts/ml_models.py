import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    RandomizedSearchCV,
    KFold
)
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

INPUT_FILE = r"C:\Users\Rafi\Documents\Claude\Projects\Capstone Project\soccer_recruitment_dataset.csv"
OUTPUT_DIR = r"C:\Users\Rafi\Documents\Claude\Projects\Capstone Project\eda_figures"
MODEL_DIR  = r"C:\Users\Rafi\Documents\Claude\Projects\Capstone Project"


# LOAD DATASET
df = pd.read_csv(INPUT_FILE)
# Remove players without a market value
df = df[df["Market_Value_EUR"].notna()].copy()
# Convert age to numeric
df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
# Create log-transformed market value
y = np.log1p(df["Market_Value_EUR"])


# FEATURE ENGINEERING
# Create the player's main position
df["PosGrp"] = df["Position_FBref"].astype(str).str.split(",").str[0].str.upper()

# Calculate the years remaining on each player's contract
expiry_date = pd.to_datetime(df["Contract_Expiry"], errors="coerce")
df["Years_Left"] = (expiry_date - pd.Timestamp("2026-01-01")).dt.days / 365.25
df["Age2"] = df["Age"] ** 2

# Remove columns that won't be used for training
drop_cols = [
    "Player", "Nation", "Position_FBref", "Club", "Club_Full_Name", "Position_TM", "Detailed_Position", "Citizenship", "Contract_Expiry",
    "Preferred_Foot", "Transfermarkt_ID", "Birth_Year", "Market_Value_EUR", "Market_Value_Millions_EUR", "Highest_Market_Value_EUR"
]

# Keep only numeric features
num_features = df.drop(
    columns=[col for col in drop_cols if col in df.columns],
    errors="ignore"
).select_dtypes(include=[np.number])

# Convert categorical features into dummy variables
cat_features = pd.get_dummies(
    df[["League", "PosGrp"]],
    drop_first=True
)

# Combine all features into one dataset
X = pd.concat([num_features, cat_features], axis=1).fillna(0)
print("Feature matrix:", X.shape)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create 5-fold cross-validation
cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# CROSS VALIDATION
# Compare models using 5-fold cross-validation
print("5-Fold Cross-Validation")

# Create machine learning models
models = {
    "Linear Regression": LinearRegression(
    ),
    "Random Forest": RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    ),
    "XGBoost": XGBRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.9,
        random_state=42
    )
}

# Evaluate each model using cross-validation
for name, model in models.items():
    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="r2"
    )
    print(f"{name:18} R² = {scores.mean():.3f} (+/- {scores.std():.3f})")




# HYPERMETER TUNNING
print("Hyperparameter Tuning (RandomizedSearchCV)")
# Define the parameter grid for Random Forest
rf_grid = {
    "n_estimators": [200, 400, 600],
    "max_depth": [None, 10, 20, 30],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", 0.5, 1.0]
}
# Search for the best Random Forest parameters
rf_search = RandomizedSearchCV(
    RandomForestRegressor(random_state=42, n_jobs=-1),
    rf_grid,
    n_iter=12,
    cv=cv,
    scoring="r2",
    random_state=42,
    n_jobs=-1
)
rf_search.fit(X_train, y_train)
print("Random Forest Best Parameters:", rf_search.best_params_)
print("Cross-Validation R²:", round(rf_search.best_score_, 3))

# Define the parameter grid for XGBoost
xgb_grid = {
    "n_estimators": [300, 500, 800],
    "learning_rate": [0.03, 0.05, 0.1],
    "max_depth": [3, 5, 7],
    "subsample": [0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.9, 1.0]
}
# Search for the best XGBoost parameters
xgb_search = RandomizedSearchCV(
    XGBRegressor(random_state=42),
    xgb_grid,
    n_iter=15,
    cv=cv,
    scoring="r2",
    random_state=42,
    n_jobs=-1
)
xgb_search.fit(X_train, y_train)
print("XGBoost Best Parameters:", xgb_search.best_params_)
print("Cross-Validation R²:", round(xgb_search.best_score_, 3))



# FINAL TEST SET EVALUATION
# Evaluate a trained model
def evaluate(model_name, model):
    predicted_log = model.predict(X_test)
    # Convert predictions back to the original market value
    predicted_value = np.expm1(predicted_log)
    actual_value = np.expm1(y_test)

    mae = mean_absolute_error(actual_value, predicted_value)
    rmse = np.sqrt(mean_squared_error(actual_value, predicted_value))
    r2 = r2_score(y_test, predicted_log)

    print(
        f"{model_name:22} "
        f"MAE = €{mae/1e6:.2f}M   "
        f"RMSE = €{rmse/1e6:.2f}M   "
        f"R² = {r2:.3f}"
    )
    return {
        "name": model_name,
        "model": model,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "predictions": predicted_log
    }
print("Final Test Set Evaluation")

# Train the Linear Regression model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Evaluate all models
results = [
    evaluate("Linear Regression", linear_model),
    evaluate("Random Forest", rf_search.best_estimator_),
    evaluate("XGBoost", xgb_search.best_estimator_)
]

# Find the best-performing model
best_model = max(results, key=lambda result: result["r2"])
print(f"\nBest Model: {best_model['name']} (R² = {best_model['r2']:.3f})")




# MODEL COMPARISON CHART
# Get the model names and R² scores
model_names = [result["name"] for result in results]
r2_scores = [result["r2"] for result in results]

# bar chart
plt.figure(figsize=(7, 4))
plt.bar(
    model_names,
    r2_scores,
    color="#1f77b4",
    edgecolor="white"
)
# Display the R² value above each bar
for i, score in enumerate(r2_scores):
    plt.text(
        i,
        score + 0.01,
        f"{score:.3f}",
        ha="center"
    )

plt.xlabel("Model")
plt.ylabel("R² Score")
plt.title("Model Comparison - Market Value Prediction")

plt.ylim(0, 1)
plt.xticks(rotation=10)

plt.tight_layout()
plt.show()




# ACTUAL VS PREDICTED MARKET VALUE
# Convert the values back to millions of euros
actual_values = np.expm1(y_test) / 1e6
predicted_values = np.expm1(best_model["predictions"]) / 1e6

# Create the scatter plot
plt.figure(figsize=(6, 6))

plt.scatter(
    actual_values,
    predicted_values,
    s=12,
    alpha=0.4,
    color="#1f77b4"
)

# Draw the perfect prediction line
max_value = max(actual_values.max(), predicted_values.max())

plt.plot(
    [0, max_value],
    [0, max_value],
    "r--",
    linewidth=1,
    label="Perfect Prediction"
)

plt.xlabel("Actual Market Value (Millions of Euros)")
plt.ylabel("Predicted Market Value (Millions of Euros)")
plt.title(f"Actual vs Predicted Values ({best_model['name']})")

plt.legend()

plt.tight_layout()
plt.show()



# Actual vs Predicted Market Value
# Convert the values back to millions of euros
actual_values = np.expm1(y_test) / 1e6
predicted_values = np.expm1(best_model["predictions"]) / 1e6

# Create the scatter plot
plt.figure(figsize=(6, 6))

plt.scatter(
    actual_values,
    predicted_values,
    s=12,
    alpha=0.4,
    color="#1f77b4"
)

# Draw the perfect prediction line
max_value = max(actual_values.max(), predicted_values.max())

plt.plot(
    [0, max_value],
    [0, max_value],
    "r--",
    linewidth=1,
    label="Perfect Prediction"
)

plt.xlabel("Actual Market Value (Millions of Euros)")
plt.ylabel("Predicted Market Value (Millions of Euros)")
plt.title(f"Actual vs Predicted Values ({best_model['name']})")

plt.legend()

plt.tight_layout()
plt.show()




# Feature Importance
# Get the best tree-based model
if hasattr(best_model["model"], "feature_importances_"):
    tree_model = best_model["model"]
else:
    tree_model = rf_search.best_estimator_

# Get the top 10 most important features
feature_importance = pd.Series(
    tree_model.feature_importances_,
    index=X.columns
).sort_values(ascending=False).head(10)

# Create the bar chart
plt.figure(figsize=(7, 5))

feature_importance.iloc[::-1].plot(
    kind="barh",
    color="#9467bd",
    edgecolor="white"
)

plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("Top 10 Most Important Features")

plt.tight_layout()
plt.show()

print("\nTop 10 Most Important Features")
print(feature_importance.round(3))