# DATA CLEANING & VISUALIZATION PROJECT

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------
# 1. LOAD DATASET
# ---------------------------------------------------------

df = pd.read_csv("student_performance.csv")

print("=" * 60)
print("ORIGINAL DATASET")
print("=" * 60)

print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())


# ---------------------------------------------------------
# 2. BASIC DATA INFORMATION
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("DATA INFORMATION")
print("=" * 60)

print(df.info())

print("\nStatistical Summary:")
print(df.describe())


# ---------------------------------------------------------
# 3. CHECK MISSING VALUES
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)

print(df.isnull().sum())


# ---------------------------------------------------------
# 4. HANDLE MISSING VALUES
# ---------------------------------------------------------

# Numerical columns
numeric_columns = df.select_dtypes(include=np.number).columns

for column in numeric_columns:
    df[column] = df[column].fillna(df[column].median())

# Categorical columns
categorical_columns = df.select_dtypes(include="object").columns

for column in categorical_columns:
    df[column] = df[column].fillna(df[column].mode()[0])


print("\nMissing values after cleaning:")
print(df.isnull().sum())


# ---------------------------------------------------------
# 5. CHECK DUPLICATES
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("DUPLICATE RECORDS")
print("=" * 60)

duplicate_count = df.duplicated().sum()

print("Number of duplicate rows:", duplicate_count)


# Remove duplicates
df = df.drop_duplicates()

print("Dataset shape after removing duplicates:")
print(df.shape)


# ---------------------------------------------------------
# 6. OUTLIER DETECTION USING IQR
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("OUTLIER DETECTION")
print("=" * 60)

for column in numeric_columns:

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    outliers = df[
        (df[column] < lower_limit) |
        (df[column] > upper_limit)
    ]

    print(column, "->", len(outliers), "outliers")


# ---------------------------------------------------------
# 7. HANDLE OUTLIERS
# ---------------------------------------------------------

for column in numeric_columns:

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    # Cap extreme values
    df[column] = df[column].clip(
        lower=lower_limit,
        upper=upper_limit
    )


print("\nOutliers handled using IQR method.")


# ---------------------------------------------------------
# 8. CREATE AVERAGE MARKS
# ---------------------------------------------------------

df["Average_Marks"] = (
    df["Maths"] +
    df["Science"] +
    df["English"]
) / 3


print("\n" + "=" * 60)
print("AVERAGE MARKS")
print("=" * 60)

print(df[["Name", "Average_Marks"]])


# ---------------------------------------------------------
# 9. SUBJECT-WISE AVERAGE
# ---------------------------------------------------------

subject_average = df[
    ["Maths", "Science", "English"]
].mean()

print("\nSubject-wise Average:")
print(subject_average)


# ---------------------------------------------------------
# 10. HIGHEST PERFORMING STUDENT
# ---------------------------------------------------------

highest_student = df.loc[
    df["Average_Marks"].idxmax()
]

print("\nHighest Performing Student:")
print(highest_student["Name"])

print(
    "Average Marks:",
    round(highest_student["Average_Marks"], 2)
)


# ---------------------------------------------------------
# 11. VISUALIZATION 1
# AVERAGE MARKS BY SUBJECT
# ---------------------------------------------------------

plt.figure(figsize=(8, 5))

subject_average.plot(kind="bar")

plt.title("Average Marks by Subject")
plt.xlabel("Subject")
plt.ylabel("Average Marks")
plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig("average_marks.png")

plt.show()


# ---------------------------------------------------------
# 12. VISUALIZATION 2
# STUDENT PERFORMANCE
# ---------------------------------------------------------

plt.figure(figsize=(10, 5))

plt.bar(
    df["Name"],
    df["Average_Marks"]
)

plt.title("Student Average Performance")
plt.xlabel("Student")
plt.ylabel("Average Marks")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig("student_performance.png")

plt.show()


# ---------------------------------------------------------
# 13. VISUALIZATION 3
# ATTENDANCE VS MARKS
# ---------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.scatter(
    df["Attendance"],
    df["Average_Marks"]
)

plt.title("Attendance vs Average Marks")
plt.xlabel("Attendance (%)")
plt.ylabel("Average Marks")

plt.tight_layout()

plt.savefig("attendance_vs_marks.png")

plt.show()


# ---------------------------------------------------------
# 14. VISUALIZATION 4
# MARKS DISTRIBUTION
# ---------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.hist(
    df["Average_Marks"],
    bins=5
)

plt.title("Distribution of Average Marks")
plt.xlabel("Average Marks")
plt.ylabel("Number of Students")

plt.tight_layout()

plt.savefig("marks_distribution.png")

plt.show()


# ---------------------------------------------------------
# 15. VISUALIZATION 5
# CORRELATION HEATMAP
# ---------------------------------------------------------

plt.figure(figsize=(8, 6))

correlation = df[
    [
        "Maths",
        "Science",
        "English",
        "Attendance",
        "Average_Marks"
    ]
].corr()

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")

plt.tight_layout()

plt.savefig("correlation_heatmap.png")

plt.show()


# ---------------------------------------------------------
# 16. SAVE CLEANED DATASET
# ---------------------------------------------------------

df.to_csv(
    "cleaned_student_performance.csv",
    index=False
)

print("\n" + "=" * 60)
print("PROJECT COMPLETED")
print("=" * 60)

print("Cleaned dataset saved as:")
print("cleaned_student_performance.csv")

print("\nVisual reports created:")
print("1. average_marks.png")
print("2. student_performance.png")
print("3. attendance_vs_marks.png")
print("4. marks_distribution.png")
print("5. correlation_heatmap.png")
