import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import io

# --- Step 1: Simulate CSV data ---
# In a real scenario, you would read your CSV file like this:
df = pd.read_csv('/home/niebieskirekin/Dokumenty/rownolegle/benchmark_result3.csv',delimiter=',')
# For demonstration purposes, we'll create a dummy CSV string.

# Use io.StringIO to treat the string as a file
# df = pd.read_csv(io.StringIO(csv_data))

# --- Step 2: Extract data for plotting ---
# threads_per_block = df['threads_per_block']
# data_size = df['data_size']
# real_time = df['real_time']

# # --- Step 3: Create the 3D plot ---
# fig = plt.figure(figsize=(10, 7)) # Create a new figure
# ax = fig.add_subplot(111, projection='3d') # Add a 3D subplot to the figure

# # Plot the 3D line graph
# # 'x', 'y', and 'z' arguments correspond to the data for each axis.
# # 'label' sets the label for the line in the legend.
# ax.plot(threads_per_block, data_size, real_time, label='Performance Data')

# # Set labels for each axis
# ax.set_xlabel('Threads Per Block')
# ax.set_ylabel('Data Size')
# ax.set_zlabel('Real Time (s)')

# # Set a title for the plot
# ax.set_title('3D Line Graph of Performance Metrics')

# # Add a legend to distinguish the plotted line
# ax.legend()

# # Optional: Adjust the view angle if needed
# # ax.view_init(elev=20, azim=-45)

# # Show the plot
# plt.show()

# --- Step 2: Prepare the plot ---
plt.figure(figsize=(12, 8)) # Create a new figure with a good size



df['threads_per_block'] = [ int(x.split('/')[2]) for x in df['name'] ]
df['data_size'] =[ int(x.split('/')[1]) for x in df['name'] ]
df['real_time'] = df['real_time'] / 1000000

# Get unique values for 'threads_per_block' to create separate series
unique_threads_per_block = df['threads_per_block'].unique()

# Sort the unique values for better legend order
unique_threads_per_block.sort()

# --- Step 3: Plot data for each 'threads_per_block' series ---
for threads in unique_threads_per_block:
    # Filter the DataFrame for the current 'threads_per_block' value
    subset_df = df[df['threads_per_block'] == threads]

    # Plot 'real_time' vs 'data_size' for this subset
    # Use a clear label for the legend
    plt.plot(subset_df['data_size'], subset_df['iterations'], marker='o', label=f'Threads/Block: {threads}')

# --- Step 4: Add plot enhancements ---
plt.xlabel('Rozmiar danych wejściowych (Bajty)') # Label for the X-axis
plt.ylabel('Liczba iteracji') # Label for the Y-axis
plt.title('(CUDA) Liczba iteracji a rozmiar danych wejściowych dla różnych ustawień Threads Per Block') # Main title
# plt.xscale('log') # Use a logarithmic scale for Data Size if values span a wide range
plt.grid(True, which="both", ls="--", c='0.7') # Add a grid for readability
plt.legend(title='Threads Per Block') # Add a legend to identify each line
plt.tight_layout() # Adjust layout to prevent labels from overlapping

# Show the plot
plt.show()