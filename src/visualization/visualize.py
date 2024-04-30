import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("../../data/raw/ratings_small.csv")

# Display the first few rows of the dataset
print(df.head())

# Descriptive statistics for numerical features
print(df.describe())

# Check for missing values in the dataset
print(df.isnull().sum())

# Distribution of ratings
plt.figure(figsize=(8, 6))
sns.countplot(x='rating', data=df)
plt.title('Distribution of Ratings')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.show()

# Number of ratings per user
ratings_per_user = df.groupby('userId')['rating'].count()
plt.figure(figsize=(8, 6))
sns.histplot(ratings_per_user, bins=30, kde=False)
plt.title('Number of Ratings per User')
plt.xlabel('Number of Ratings')
plt.ylabel('Number of Users')
plt.show()

# Number of ratings per movie
ratings_per_movie = df.groupby('movieId')['rating'].count()
plt.figure(figsize=(8, 6))
sns.histplot(ratings_per_movie, bins=30, kde=False)
plt.title('Number of Ratings per Movie')
plt.xlabel('Number of Ratings')
plt.ylabel('Number of Movies')
plt.show()

# Convert Unix timestamps to readable dates
df['date'] = pd.to_datetime(df['timestamp'], unit='s')

# Ratings over time
plt.figure(figsize=(10, 6))
df.groupby(df['date'].dt.year)['rating'].mean().plot(kind='line')
plt.title('Average Rating Over Time')
plt.xlabel('Year')
plt.ylabel('Average Rating')
plt.show()

# User activity analysis
user_activity = df.groupby('userId')['rating'].count()
plt.figure(figsize=(10, 6))
sns.boxplot(x=user_activity)
plt.title('User Activity Distribution')
plt.xlabel('Number of Ratings per User')
plt.show()