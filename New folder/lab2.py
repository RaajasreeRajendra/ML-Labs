import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data.csv")

# Experiment 1
print("Missing values:")
print(df.isnull())
print("\nTotal missing values per column:")
print(df.isnull().sum())

# Experiment 2
print("\nDataset size before cleaning:", df.shape)
clean_df = df.dropna()
print("Dataset size after cleaning:", clean_df.shape)

# Experiment 3
df_filled = df.fillna(df.mean(numeric_only=True))
print("\nAfter filling missing values:")
print(df_filled)

# Experiment 4
print("\nData types before:")
print(df.dtypes)

# Example conversion
df["Age"] = df["Age"].astype("float")

print("\nData types after:")
print(df.dtypes)

# Experiment 5
df = df.rename(columns={"Name":"Student_Name"})
print("\nRenamed columns:")
print(df.head())

# Experiment 6
df = df.replace("NA","Unknown")
print("\nAfter replacing incorrect values:")
print(df.head())

# Experiment 7
df["Marks"].plot(kind="line",title="Marks Trend")
plt.show()

df.groupby("Student_Name")["Marks"].mean().plot(kind="bar",title="Marks Comparison")
plt.show()

# Experiment 8
plt.scatter(df["Age"],df["Marks"])
plt.xlabel("Age")
plt.ylabel("Marks")
plt.title("Age vs Marks")
plt.show()

# Experiment 9
df["Marks"].plot(kind="hist",bins=10,title="Marks Distribution")
plt.show()

# Experiment 10
df.boxplot(column="Marks")
plt.title("Marks Boxplot")
plt.show()
