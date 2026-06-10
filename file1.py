import pandas as pd
import numpy as np

# ----------------------------
# Step 1: Create students_data.csv
# ----------------------------
data = {
    "Name": ["Ali", "Saram", "Ahmed", "Hadi", "Omar", "Noman", "Raza", "Zia"],
    "Age": [20, 21, 22, 20, 23, 21, 22, 19],
    "Marks": [85, 90, 78, np.nan, 65, 88, 70, 95]
}

# Save dataset
df = pd.DataFrame(data)
df.to_csv("students_data.csv", index=False)

# ----------------------------
# Step 2: Read dataset using Pandas
# ----------------------------
df = pd.read_csv("students_data.csv")
print("Original Data:")
print(df)

# Replace missing values in Marks with 0
df["Marks"] = df["Marks"].fillna(0)
print("\nAfter filling missing values:")
print(df)

# ----------------------------
# Step 3: Define assign_grade function
# ----------------------------
def assign_grade(marks, passing=70):
    if marks >= 85:
        return "A"
    elif passing <= marks < 85:
        return "B"
    else:
        return "C"

# Apply grades
df["Grade"] = df["Marks"].apply(assign_grade)
print("\nWith Grades:")
print(df)

# ----------------------------
# Step 4: Extract Marks as NumPy array
# ----------------------------
marks_array = df["Marks"].to_numpy()
print("\nMarks as NumPy array:")
print(marks_array)

# ----------------------------
# Step 5: Normalize function
# ----------------------------
def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

normalized_marks = normalize(marks_array)

# ----------------------------
# Step 6: Create new DataFrame
# ----------------------------
norm_df = pd.DataFrame({
    "Original_Marks": marks_array,
    "Normalized_Marks": normalized_marks
})
print("\nNew DataFrame with Normalized Marks:")
print(norm_df)

# ----------------------------
# Step 7: Save to CSV
# ----------------------------
norm_df.to_csv("normalized_marks.csv", index=False)

# ----------------------------
# Step 8: Read back and display
# ----------------------------
read_back = pd.read_csv("normalized_marks.csv")
print("\nContents of normalized_marks.csv:")
print(read_back)
