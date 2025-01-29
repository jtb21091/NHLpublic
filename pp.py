import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import scipy.stats as stats

# Load the Excel file
file_path = "Summary.xlsx"  # Update this to the correct file path
xls = pd.ExcelFile(file_path)

# Load the "Summary" sheet into a DataFrame
df = pd.read_excel(xls, sheet_name="Summary")

# Extract PP% and P% columns for analysis
x = df["PP%"].values.reshape(-1, 1)  # Independent variable
y = df["P%"].values.reshape(-1, 1)  # Dependent variable

# Perform linear regression
model = LinearRegression()
model.fit(x, y)
y_pred = model.predict(x)

# Calculate R-squared
r2 = r2_score(y, y_pred)

# Calculate p-value
slope, intercept, r_value, p_value, std_err = stats.linregress(df["PP%"], df["P%"])

# Create the plot
plt.figure(figsize=(12, 8))
plt.scatter(df["PP%"], df["P%"], color="blue", label="Data points")
plt.plot(df["PP%"], y_pred, color="red", label=f"Trendline (R²={r2:.2f})")

# Annotate team names
for i, team in enumerate(df["Team"]):
    plt.text(df["PP%"][i], df["P%"][i], team, fontsize=8, ha="right")

# Add regression formula under R-squared on the plot
plt.text(
    max(df["PP%"])*0.7, 
    max(df["P%"])*0.85, 
    f"y = {slope:.5f}x + {intercept:.5f}", 
    fontsize=10, 
    color="red"
)

plt.title("P% vs PP%")
plt.xlabel("PP%")
plt.ylabel("P%")
plt.legend()
plt.grid()

# Save the plot as a PNG
output_file = "pp_vs_p_percent_analysis.png"
plt.savefig(output_file, dpi=300)
plt.show()

print(f"Analysis complete. R-squared: {r2:.2f}, p-value: {p_value:.10f}")
print(f"Plot saved as {output_file}")
