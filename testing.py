import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # Import the 3D plotting module
import seaborn as sns

# After clustering
df = pd.read_csv("output.csv")
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')  # Create a 3D subplot
ax.scatter(df["x"], df["y"], df["z"], c=df["c"], cmap='viridis')  # Scatter plot with color based on "c" column
ax.set_xlabel("Danceability")
ax.set_ylabel("Energy")
ax.set_zlabel("Valence")
plt.title("The Genre Reveal Party")
# plt.show() # only able to use with a jupyter notebook or similar
# Save picture of 3D plot as [k]-output-[epochs].png
plt.savefig("5-output-5000.png")
