# Employee-Performance-Analytics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
data = pd.read_csv("/content/sample_data/Employee_Dataset.csv")
data.head()

# **Understanding the Dataset**
# Check the dimensions of the dataset
print("Dimensions of the dataset (rows, columns):", data.shape)

# Get a summary of the data types, non-null counts, and memory usage
print("\nData Summary:")
print(data.info())

# Check the column names
print("\nColumn Names:")
print(data.columns)

# Rename the column names
data.rename(columns={'employee_id':'Employee_ID', 'department':'Department', 'region':'Region', 'education':'Education', 'gender':'Gender',
       'recruitment_channel':'Recruitment_Channel', 'no_of_trainings':'No_of_Trainings', 'age':'Age', 'previous_year_rating':'Previous_Year_Rating',
       'length_of_service':'Length_of_Service', 'KPIs_met_more_than_80':'KPIs_Met_More_Than_80', 'awards_won':'Awards_Won',
       'avg_training_score':'Avg_Training_Score'},inplace=True)

# **Data Cleaning**
# Check null values
print("\nHow many null values present in the column:")
print(data.isnull().sum())
# As we see there are two null values present in the dataset.

# fill the null values
data['Previous_Year_Rating'].fillna(data['Previous_Year_Rating'].mean(), inplace=True)
data['Education'].fillna(method='bfill', inplace=True)

# check duplicate value
data.duplicated().sum()
# There are 2 deplicate values present in the dataset.

# **Descrptive Statistics**
data.describe().transpose()

# **Data Visualisation**
# 1. Plot Histograms or Boxplots for Numerical Variables.
# Assuming you have the DataFrame 'data' with numerical columns
numerical_columns = ['Age', 'No_of_Trainings', 'Previous_Year_Rating', 'Length_of_Service', 'Avg_Training_Score']
# Plot histograms for numerical variables
for column in numerical_columns:
    plt.figure(figsize=(8, 6))
    sns.histplot(data[column], kde=True)
    plt.title(f'Histogram of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()

# 2. Plot boxplots for numerical variables.
for column in numerical_columns:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=data[column])
    plt.title(f'Boxplot of {column}')
    plt.xlabel(column)
    plt.show()

# 3. Use Bar Charts or Pie Charts for Categorical Variables.
# Assuming you have the DataFrame 'data' with categorical columns
categorical_columns = ['Department', 'Education', 'Gender', 'Recruitment_Channel', 'KPIs_Met_More_Than_80', 'Awards_Won']
# Plot bar charts for categorical variables
for column in categorical_columns:
    plt.figure(figsize=(8, 6))
    sns.countplot(x=column, data=data, palette='viridis')
    plt.title(f'Bar Chart of {column}')
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.show()

# 4. Plot pie charts for categorical variables.
for column in categorical_columns:
    plt.figure(figsize=(8, 6))
    data[column].value_counts().plot(kind='pie', autopct='%1.1f%%', colormap='Paired')
    plt.title(f'Pie Chart of {column}')
    plt.ylabel('')
    plt.show()

# 5. Create Scatter Plots or Line Plots for Interesting Relationships.
# Assuming you want to create scatter plots or line plots between two numerical variables
interesting_relationships = [('Age', 'Avg_Training_Score'), ('Length_of_Service', 'No_of_Trainings')]
# Create scatter plots
for x, y in interesting_relationships:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=x, y=y, data=data)
    plt.title(f'Scatter Plot: {x} vs {y}')
    plt.xlabel(x)
    plt.ylabel(y)
    plt.show()

# 6. Awards won by each gender.
# Assuming you have the DataFrame 'data'
awards_by_gender = data[data['Awards_Won'] == 1]['Gender'].value_counts()
# Create a bar plot to visualize awards won by each gender
plt.figure(figsize=(8, 6))
awards_by_gender.plot(kind='bar', color=['blue', 'pink'], alpha=0.7)
plt.title('Awards Won by Gender')
plt.xlabel('Gender')
plt.ylabel('Number of Awards Won')
plt.xticks(rotation=0)
plt.show()

# 7. Award won by each department.
# Assuming you have the DataFrame 'data'
awards_by_department = data[data['Awards_Won'] == 1]['Department'].value_counts()
# Create a count plot (bar plot) to visualize awards won by each department
plt.figure(figsize=(10, 6))
sns.countplot(x='Department', data=data[data['Awards_Won'] == 1], palette='Set2')
plt.title('Awards Won by Department')
plt.xlabel('Department')
plt.ylabel('Number of Awards Won')
plt.xticks(rotation=45, ha='right')  # Rotate and align department names for better readability
plt.show()


# 8. Awards won by each Age.
# Assuming you have the DataFrame 'data'
awards_by_Age = data[data['Awards_Won'] == 1]['Age'].value_counts()
# Create a count plot (bar plot) to visualize awards won by each Age
plt.figure(figsize=(10, 6))
sns.countplot(x='Age', data=data[data['Awards_Won'] == 1], palette='Set2')
plt.title('Awards Won by Age')
plt.xlabel('Age')
plt.ylabel('Number of Awards Won')
plt.xticks(rotation=45, ha='right')  # Rotate and align department names for better readability
plt.show()
# Assuming you have the DataFrame 'hr'
highest_age_awards_won = data[data['Awards_Won'] == 1]['Age'].max()
print("Highest Age of Employees who won awards:", highest_age_awards_won)

# 9. Visualize training score by gender.
# Assuming you have the DataFrame 'data'
avg_training_score_by_gender = data.groupby('Gender')['Avg_Training_Score'].mean()
# Create a pie chart to visualize average training score by gender
plt.figure(figsize=(8, 6))
plt.pie(avg_training_score_by_gender, labels=avg_training_score_by_gender.index, autopct='%1.1f%%', colors=['purple', 'orange'])
plt.title('Training Score by Gender')
plt.show()

# 10. Visualize training score by department.
# Assuming you have the DataFrame 'data'
avg_training_score_by_department = data.groupby('Department')['Avg_Training_Score'].mean()
# Create a pie chart to visualize average training score by department
plt.figure(figsize=(8, 6))
plt.pie(avg_training_score_by_department, labels=avg_training_score_by_department.index, autopct='%1.1f%%', colors=['purple', 'orange','green','yellow','blue','pink','brown','skyblue','red'])
plt.title('Training Score by Department')
plt.show()

# 11. Correlation analysis and visualize the correlation matrix using a heatmap.
# Compute the correlation matrix
correlation_matrix = data[numerical_columns].corr()
# Create a heatmap to visualize the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

data.to_csv('cleaned_data.csv', index=False)
