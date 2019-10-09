#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

#%%
#read csv
movies = pd.read_csv('data/movies.csv')
ratings = pd.read_csv('data/ratings.csv')
ratings
#%%merge dataframes on movieId 
data = pd.merge(left=movies, right=ratings ,on='movieId')

#%%
data.head()

#%%
data.isnull().sum()

#%%
data.shape

#%%
data['movieId'].nunique()

#%%
years =[]

for title in data['title']:
    year_subset = title[-5:-1]
    try:
        years.append(int(year_subset))
    except:
        years.append(9999)
data['moviePublishYears'] = years
data
print(len(data[data['moviePublishYears'] == 9999]))
#%%
data = data[data['moviePublishYears'] != 9999]
len(data)

#%%
def make_histogram(dataset, attribute, bins=25, bar_color='#3498db', edge_color='#2980b9', title='Title', xlab='X', ylab='Y', sort_index=False):
    if attribute == 'moviePublishYears':
        dataset = dataset[dataset['moviePublishYears'] != 9999]
        
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title(title, fontsize=24, pad=20)
    ax.set_xlabel(xlab, fontsize=16, labelpad=20)
    ax.set_ylabel(ylab, fontsize=16, labelpad=20)
    
    plt.hist(dataset[attribute], bins=bins, color=bar_color, ec=edge_color, linewidth=2)
    
    plt.xticks(rotation=45)
    
make_histogram(data, 'moviePublishYears', title='Movies Published per Year', xlab='Year', ylab='Counts')



#%%
make_histogram(data, 'rating', title='Ratings for movies', xlab='rating', ylab='Counts')

#%%
plt.hist(data['rating'])


#%%
genre_df = pd.DataFrame(data['genres'].str.split('|').tolist(), index=data['movieId']).stack()
genre_df = genre_df.reset_index([0, 'movieId'])
genre_df.columns = ['movieId', 'Genre']

#%%
genre_df.head()

#%%
def make_bar_chart(dataset, attribute, bar_color='#3498db', edge_color='#2980b9', title='Title', xlab='X', ylab='Y', sort_index=False):
    if sort_index == False:
        xs = dataset[attribute].value_counts().index
        ys = dataset[attribute].value_counts().values
    else:
        xs = dataset[attribute].value_counts().sort_index().index
        ys = dataset[attribute].value_counts().sort_index().values
        
    
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title(title, fontsize=24, pad=20)
    ax.set_xlabel(xlab, fontsize=16, labelpad=20)
    ax.set_ylabel(ylab, fontsize=16, labelpad=20)
    
    plt.bar(x=xs, height=ys, color=bar_color, edgecolor=edge_color, linewidth=2)
    plt.xticks(rotation=45)
    
    
make_bar_chart(genre_df, 'Genre', title='Most Popular Movie Genres', xlab='Genre', ylab='Counts')


#%%
values = defaultdict(list)
for ind, row in data.iterrows():
    for genre in row['genres'].split('|'):
        values[genre].append(row['rating'])

#%%
genre_lst, rating_lst = [], []
for key, item in values.items():
    if key not in [0, 1]:
        genre_lst.append(key)
        rating_lst.append(np.mean(item))

#%%
genres_with_ratings = pd.DataFrame([genre_lst, rating_lst]).T
genres_with_ratings.columns = ['Genre', 'Mean_Rating']

#%%
fig, ax = plt.subplots(figsize=(14, 7))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_title('Rating by genre', fontsize=24, pad=20)
ax.set_xlabel('Genre', fontsize=16, labelpad=20)
ax.set_ylabel('Ratings', fontsize=16, labelpad=20)
plt.bar(x=genres_with_ratings['Genre'], height=genres_with_ratings['Mean_Rating'], color='#3498db', edgecolor='#2980b9', linewidth=2)
plt.xticks(rotation=45)


#%%
num_ratings = pd.DataFrame(data.groupby('movieId').count()['rating'])


#%%
data = pd.merge(left=data, right=num_ratings, on='movieId')
data.rename(columns={'rating_x': 'rating', 'rating_y': 'numRatings'}, inplace=True)
#%%
data.sort_values(by='numRatings', ascending=False).drop_duplicates('movieId')[:10]
data

#%%
ratings_df = pd.DataFrame()
ratings_df['Mean_Rating'] = data.groupby('title')['rating'].mean().values
ratings_df['Num_Ratings'] = data.groupby('title')['rating'].count().values


fig, ax = plt.subplots(figsize=(14, 7))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_title('Rating vs. Number of Ratings', fontsize=24, pad=20)
ax.set_xlabel('Rating', fontsize=16, labelpad=20)
ax.set_ylabel('Number of Ratings', fontsize=16, labelpad=20)

plt.scatter(ratings_df['Mean_Rating'], ratings_df['Num_Ratings'], alpha=0.5)

#%%
