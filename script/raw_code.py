import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.ticker import FuncFormatter
from datetime import datetime
import statsmodels.formula.api as smf
from statsmodels.iolib.summary2 import summary_col
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score


df = pd.read_csv('btc.csv')


def convert_to_numeric(value):
    if isinstance(value, str):
        # Handle empty or invalid values
        if value == '--' or value.strip() == '':
            return None
            
        # Remove commas
        value = value.replace(',', '')
        
        try:
            # Remove 'M' if present
            if value.endswith('M'):
                value = value[:-1]
            
            # Convert string to float
            num = float(value)
            # Multiply by 1 million
            return num * 1_000_000
        except ValueError:
            return None
    return value


# Example usage with pandas DataFrame
import pandas as pd


# Assuming your DataFrame is called 'df'
# Apply to all columns except 'Time (UTC)'
numeric_columns = ['BTCO', 'IBIT', 'BITB', 'FBTC', 'GBTC', 'HODL', 'BRRR', 'ZEBC', 'ARKB', 'BTCW', 'TOTAL']
for col in numeric_columns:
    df[col] = df[col].apply(convert_to_numeric)


df.loc[df['BITB'] == 700000000000, 'BITB'] = 700000


# List of columns to calculate the mean
columns = ['BTCO', 'IBIT', 'BITB', 'FBTC', 'GBTC', 'HODL', 'BRRR', 'ZEBC', 'ARKB', 'BTCW', 'TOTAL']


# Calculate the mean for the specified columns, excluding the row labeled 'Total'
mean_values = df[df.index != 'Total'][columns].mean()


# Display the mean values
print(mean_values)


# List of columns to calculate the mean
columns = ['BTCO', 'IBIT', 'BITB', 'FBTC', 'GBTC', 'HODL', 'BRRR', 'ZEBC', 'ARKB', 'BTCW', 'TOTAL']

# Calculate the mean for the specified columns, excluding the row labeled 'Total'
mean_values = df[df.index != 'Total'][columns].mean()

# Display the mean values
print(mean_values)


import matplotlib.pyplot as plt

# Assuming these are the mean values you calculated
mean_values = {
    'BTCO': 9452873.56,
    'IBIT': 298643137.25,
    'BITB': 427042609.09,
    'FBTC': 113209340.66,
    'GBTC': -236108994.08,
    'HODL': 12625000.0,
    'BRRR': 16369696.97,
    'ZEBC': 12439436.62,
    'ARKB': 35763125.0,
    'BTCW': 922531.91,
    'TOTAL': 214755736.04
}

# Extract data for plotting
columns = list(mean_values.keys())
values = list(mean_values.values())

divided_values = [value / 2 for value in values]

# Create the bar chart
plt.figure(figsize=(10, 6))
plt.bar(columns,[v / 1_000_000 for v in divided_values], color='skyblue')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
plt.xlabel('Columns')
plt.ylabel('Mean Values(Million)')
plt.title('Mean Values Bar Chart')

formatter = FuncFormatter(lambda x, pos: f'{x:.0f}M' if x != 0 else '0')
plt.gca().yaxis.set_major_formatter(formatter)

plt.grid(axis='y', linestyle='--', alpha=0.6)  # Add horizontal grid lines for clarity
plt.tight_layout()


# Show the plot
plt.show()


df['Time (UTC)'] = pd.to_datetime(df['Time (UTC)'], errors='coerce')

df['Time (UTC)'] = pd.to_datetime(df['Time (UTC)'])
plt.figure(figsize=(15, 8))

# Columns to plot (excluding Time (UTC))
columns_to_plot = ['BTCO', 'IBIT', 'BITB', 'FBTC', 'GBTC', 'HODL', 'BRRR', 'ZEBC', 'ARKB', 'BTCW']

for column in columns_to_plot:
    # Convert data to numeric values
    values = df[column]
    plt.scatter(df['Time (UTC)'], values, label=column, alpha=0.7)

    plt.title('Value Distribution Over Time', fontsize=14, pad=20)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Values', fontsize=12)

plt.xticks(rotation=45)

# Add legend
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Add grid for better readability
plt.grid(True, linestyle='--', alpha=0.3)

# Adjust layout to prevent label cutoff
#plt.tight_layout()

# Show the plot
plt.show()


df['TOTAL_accumulated'] = df['TOTAL'].cumsum()
df['TOTAL_IBIT'] = df['IBIT'].cumsum()

df.rename(columns={'btc price': 'btc_price'}, inplace=True)

# Create the model with the new column name
mod_1 = smf.ols('btc_price ~ TOTAL_accumulated', data=df)
res_1 = mod_1.fit()
print(res_1.summary())


#import matplotlib.pyplot as plt

plt.scatter(df['TOTAL_accumulated'], df['btc_price'], color = 'orange', marker = '+')
plt.xlabel('TOTAL_accumulated')
plt.ylabel('btc_price')
plt.title('Scatter plot of TOTAL_accumulated vs. btc_price')
plt.show()



# Calculate category totals and percentages
treemap_data = df[['BTCO', 'IBIT', 'BITB', 'FBTC', 'HODL', 'BRRR', 'ZEBC', 'ARKB', 'BTCW']].sum().reset_index()
treemap_data.columns = ['Category', 'Total']

overall_total = treemap_data['Total'].sum()

treemap_data['Percentage'] = (treemap_data['Total'] / overall_total) * 100

# Adjust percentages to avoid rounding errors (optional)
treemap_data['Percentage'] = treemap_data['Percentage'].apply(lambda x: round(x, 2))

# Create labels with larger font size for percentages
treemap_data['Label'] = treemap_data['Category'] + ': <span style="font-size: 18pt;">' + treemap_data['Percentage'].astype(str) + '%</span>'

# Create the treemap
fig = px.treemap(treemap_data, 
                 path=['Label'], 
                 values='Total',
                 title='Treemap of Total Values by Category',
                 color='Category', 
                 color_discrete_sequence=px.colors.qualitative.Bold)


fig.update_layout(
    width=800,  # Set a larger width
    height=600   # Set a larger height
)


# Show the plot
fig.show()


df['btc_price_change'] = (df['btc_price'].diff() > 0).astype(int)
# Drop the first row with NaN because of diff()
df = df.dropna()

# Define X (feature) and y (target)
X = df[['TOTAL','Total volume']]
y = df['btc_price_change']

# Step 2: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the Decision Tree model
clf = DecisionTreeClassifier()

# Train the model
clf.fit(X_train, y_train)

# Step 3: Make predictions and evaluate the model
y_pred = clf.predict(X_test)

# Print evaluation results
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Create a binary outcome (1 if btc_price goes up, 0 if it goes down)
df['btc_price_change'] = (df['btc_price'].diff() > 0).astype(int)

# Drop the first row with NaN because of diff()
df = df.dropna()

# Define X (features) and y (target) - including both 'TOTAL' and 'Total volume'
X = df[['TOTAL', 'Total volume']]
y = df['btc_price_change']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train the SVM model
model = SVC(kernel='poly')  # You can change the kernel to 'rbf', 'poly', etc.
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate the accuracy
accuracy_svc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy_svc * 100:.2f}%")

# Print classification report
print("\nClassification Report:\n", classification_report(y_test, y_pred))





