import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # Import the 3D plotting module
import seaborn as sns

def visualize(output_csv, output_png):
    # After clustering
    df = pd.read_csv(output_csv)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')  # Create a 3D subplot
    ax.scatter(df["x"], df["y"], df["z"], c=df["c"], cmap='viridis')  # Scatter plot with color based on "c" column
    ax.set_xlabel("Danceability")
    ax.set_ylabel("Energy")
    ax.set_zlabel("Valence")
    plt.title("The Genre Reveal Party")
    # plt.show() # only able to use with a jupyter notebook or similar
    # Save picture of 3D plot as [k]_output_[type]_[epochs].png
    plt.savefig(output_png)

# visualize([output_csv="output_[HPC_type].csv"], [output_png="[k]_output_[type]_[epochs].png"])
visualize("output_serial.csv", "5_output_serial_500.png")
visualize("output_shared_cpu.csv", "5_output_serial_500.png")
visualize("output_shared_gpu.csv", "5_output_serial_500.png")
visualize("output_distributed_cpu.csv", "5_output_serial_500.png")
visualize("output_distributed_gpu.csv", "5_output_serial_500.png")
