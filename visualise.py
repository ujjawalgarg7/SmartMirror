import pandas as pd
import plotly.express as px
import altair as alt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load dataset
file_path = 'color_data.csv'  # Ensure the CSV file exists in the directory
df = pd.read_csv(file_path)

# Assign unique colors to each color family
unique_families = df['Color_Family'].unique()
color_map = {family: px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)] 
             for i, family in enumerate(unique_families)}
df['Color'] = df['Color_Family'].map(color_map)

# Convert RGB values to float scale (0-1)
df[['R', 'G', 'B']] = df[['R', 'G', 'B']].astype(int)

# Plotly 3D Scatter Plot
fig_3d = px.scatter_3d(df, x='R', y='G', z='B', color='Color_Family', 
                        title='Color Clusters in RGB Space', opacity=0.7)
fig_3d.show()

# Altair 2D Scatter Plots
scatter_rg = alt.Chart(df).mark_circle().encode(
    x='R', y='G', color='Color_Family', tooltip=['Color_Family', 'R', 'G', 'B']
).properties(title='Red vs Green')

scatter_rb = alt.Chart(df).mark_circle().encode(
    x='R', y='B', color='Color_Family', tooltip=['Color_Family', 'R', 'G', 'B']
).properties(title='Red vs Blue')

scatter_gb = alt.Chart(df).mark_circle().encode(
    x='G', y='B', color='Color_Family', tooltip=['Color_Family', 'R', 'G', 'B']
).properties(title='Green vs Blue')

scatter_rg.display()
scatter_rb.display()
scatter_gb.display()

# Matplotlib 3D Scatter Plot as an alternative
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df['R'], df['G'], df['B'], c=df['Color'], alpha=0.7, edgecolors='k')
ax.set_xlabel('Red')
ax.set_ylabel('Green')
ax.set_zlabel('Blue')
ax.set_title('Color Clusters in RGB Space (Matplotlib)')
plt.show()



import numpy as np

# Compute Z-score
df[['R_z', 'G_z', 'B_z']] = (df[['R', 'G', 'B']] - df[['R', 'G', 'B']].mean()) / df[['R', 'G', 'B']].std()

# Identify outliers (absolute Z-score > 3)
outliers = df[(df['R_z'].abs() > 3) | (df['G_z'].abs() > 3) | (df['B_z'].abs() > 3)]
print("Outliers detected using Z-score method:\n", outliers)


# Compute IQR
Q1 = df[['R', 'G', 'B']].quantile(0.25)
Q3 = df[['R', 'G', 'B']].quantile(0.75)
IQR = Q3 - Q1

# Identify outliers
outliers = df[((df[['R', 'G', 'B']] < (Q1 - 1.5 * IQR)) | (df[['R', 'G', 'B']] > (Q3 + 1.5 * IQR))).any(axis=1)]
print("Outliers detected using IQR method:\n", outliers)


from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# Normalize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[['R', 'G', 'B']])

# Apply DBSCAN
dbscan = DBSCAN(eps=1.5, min_samples=3)
df['Outlier'] = dbscan.fit_predict(scaled_data)

# Identify outliers (DBSCAN marks them as -1)
outliers = df[df['Outlier'] == -1]
print("Outliers detected using DBSCAN:\n", outliers)


from sklearn.ensemble import IsolationForest

# Fit Isolation Forest
iso_forest = IsolationForest(contamination=0.05, random_state=42)
df['Outlier'] = iso_forest.fit_predict(df[['R', 'G', 'B']])

# Identify outliers (marked as -1)
outliers = df[df['Outlier'] == -1]
print("Outliers detected using Isolation Forest:\n", outliers)
