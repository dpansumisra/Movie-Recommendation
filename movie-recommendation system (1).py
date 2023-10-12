#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


credits = pd.read_csv('tmdb_5000_credits.csv')
movies = pd.read_csv('tmdb_5000_movies.csv')


# In[3]:


credits.head(1)


# In[4]:


movies.head(1)


# In[5]:


movies = movies.merge(credits, on='title')


# In[6]:


movies.head(1)


# In[7]:


# movie selecting criteria
# id
# genres
# keywords
# title
# overview
# cast 
# crew
movies = movies[['id','title' ,'overview' ,'genres','keywords','cast','crew']]


# In[8]:


movies.head()


# In[9]:


#  > for the further execution of recommendation system we creating a simple data set
#    which means we merging some of the columns like 'genres' , 'keywords' 'overview' , 'cast' , 'crew' and
#    after the merging we can say that tagline
#  > so basically our dataset have 3 column which are of ['id' , 'title' , 'tagline']


# In[10]:


# now finding the missing data
movies.isnull().sum()


# In[11]:


# then we fixing our missing data
movies.dropna(inplace=True)
movies.isnull().sum()


# In[12]:


# now we check about duplicacy data
movies.duplicated().sum()


# In[13]:


movies.iloc[0].genres


# In[14]:


# now pre-processing this -->
# '[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]'
#     --> into
# ['Action','Adventure','Fantasy','Sci-Fi']


# In[15]:


#        > creating an function for storing the values of pre-processing data
# def convert(obj):
#     L = []
#     for i in obj:
#         L.append(i['name'])
#     return L


# In[16]:


#    > above the function gets the error -->  error TypeError: string indices must be integers
#      because our "name" is string


#    > so we need to convert our string into integer--> we using "ast module" for removing the error


# In[17]:


import ast

def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L


# In[18]:


movies['genres'] = movies['genres'].apply(convert)
movies['genres'].head()


# In[19]:


movies['keywords'] = movies['keywords'].apply(convert)
movies['keywords'].head()


# In[20]:


#  now lets come on the cast. so, in every movie have lot of cast members like main actor and side actors but for the
#  recommendation system we need to check only main character. And the main character of the movies is may be 3 name of the
#  movie of we just selecting the 3 main character of every movie 


# In[21]:


def convert3(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):
        if (counter != 3):
            L.append(i['name'])
            counter += 1
        else:
            break
    return L


# In[22]:


movies['cast'] = movies['cast'].apply(convert3)
movies['cast'].head()


# In[23]:


# this is our new dataset, 
movies.head()


# In[24]:


#  > Now we fixing our crew, so everyone likes the movies of the specific crew we also say that according to "Director"
#    we selecting the movies

def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L


# In[25]:


movies['crew'] = movies['crew'].apply(fetch_director)
movies['crew'].head()


# In[26]:


movies['overview'] = movies['overview'].apply(lambda x : x.split())
movies['overview'].head()


# In[27]:


movies.head()


# In[28]:


#    > now we removing the space between the 'genres', 'keywords', 'cast', 'crew'
#    > because suppose some wants the movie of 'tom holland' but our recommedated system doesnt understand due to
#      presence of 'tom cruise'.... so we removing the space for that


# In[29]:


movies['genres'] = movies['genres'].apply(lambda x : [i.replace(" ",'') for i in x])
movies['genres'].head()


# In[30]:


movies['keywords'].apply(lambda x : [i.replace(" ",'') for i in x])
movies['keywords'].head()


# In[31]:


movies['genres'] = movies['genres'].apply(lambda x : [i.replace(" ",'') for i in x])
movies['genres'].head()


# In[32]:


movies['crew'] = movies['crew'].apply(lambda x : [i.replace(" ",'') for i in x])
movies['crew'].head()


# In[33]:


movies.head()


# In[34]:


movies['tagline'] = movies['genres'] + movies['overview'] + movies['keywords'] + movies['cast'] + movies['crew'] 


# In[35]:


movies.head()


# In[36]:


new_df = movies[['id' , 'title' , 'tagline']]
new_df.head()


# In[37]:


#  'tagline' is a list so we conveting into string

new_df['tagline'] = new_df['tagline'].apply(lambda x:" ".join(x))


# In[38]:


new_df['tagline'].head()


# In[39]:


new_df.head()


# In[40]:


# so how to look our 'tagline'

new_df['tagline'][0]


# In[41]:


new_df['tagline'] = new_df['tagline'].apply(lambda x: x.lower())


# In[42]:


new_df['tagline'].head()


# In[43]:


#  converting text into vectors
#     text --> vector
#  1). concatenating all the taglines 
#           tagline[0] + tagline[1] + tagline[2] + ....... 
#     and finding most frequent words (and we selecting maximum 5000 words)
#           w1|w2|w3|w4|w5|........|w5000
#  2). then finding the distance between the points in the graphical methods i.e. if we suggesting some movie then
#      we go to check the nearest point of the given point(what the suggested movie we need to pick)


# In[44]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 5000, stop_words = 'english')


# In[45]:


#  > max_features = for maximum words which we selecting from tagline
#  > stop_words = for removing stop words from tagline like in,is,are,on


# In[46]:


# CountVectorizer return the sparse matrix
# <4806x5000 sparse matrix of type '<class 'numpy.int64'>'
#   	with 161516 stored elements in Compressed Sparse Row format>
#  

# so we use numpy array for storing the value of tagline

cv.fit_transform(new_df['tagline']).toarray().shape


# In[47]:


vectors = cv.fit_transform(new_df['tagline']).toarray()
vectors


# In[48]:


vectors[0]        # this is the first movie


# In[49]:


cv.get_feature_names_out()


# In[50]:


#     > then we facing some kind of issue like, somewhere we get 'action' and 'action' , so due to our 
#       recommendation system count differently both of them but the meaning of both are very much similar
 
#     > so we import a python library "nltk" nltk = is based on NLP
#     > we using nltk for resolving the issue with the help of "stem" function
#          stem = changes the words into root word
#              ['loved','loving','love'] ---> ['love','love','love']
        


# In[51]:


import nltk


# In[52]:


from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


# In[53]:


def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)


# In[54]:


new_df['tagline'] = new_df['tagline'].apply(stem)
new_df['tagline'].head()


# In[60]:


#     > after using "stem" function
#     > so right now there is no issue of like 'action' or 'actions' , both are same, it is consider as single words

cv.get_feature_names_out()


# In[61]:


#  for recommedation we finding the distance between the points, we have two option to finding the distance beteen point
#  1). by the Euclidean distance = it is the distance between the top-tip points of the lines.....so we cannot
#      use this method for finding the distance.
#  2). by "Cosine distance" = it is the distance by the angle between lines....so we using this method
#                             > if the angle is minimum between lines, it mean there is small distance
#                                 like '0 degree'
#                             > if the angle is large, it means there is large distance
#                                 like ' 90 degree'
#  so we using " Cosine Distance"


# In[62]:


from sklearn.metrics.pairwise import cosine_similarity


# In[63]:


similarity = cosine_similarity(vectors)


# In[64]:


similarity.shape


# the meaning of (4806, 4806) ---> (distance the movie with other 4806 movies, distance the movie with other 4806 movies)


# In[65]:


similarity[0]



#  similarity[0] means ---> the similarity of the first movie with the first movie are exact same,


# In[66]:


similarity[1]


#  similarly with [1]  ---> the movie of second index is exact with second movie and have 
#                           differnt value for different movies


# In[67]:


def recommend(movie):
    movie_index =  new_df[new_df['title'] == movie].index[0]
    distance = similarity[movie_index]
    movies_list = sorted(list(enumerate(distance)),reverse =True , key = lambda x: x[1])[1:6]
    
    for i in movies_list:
#        print(i[0])
        print(new_df.iloc[i[0]].title)


# In[68]:


#                sorted(list(enumerate(similarity)),reverse =True , key = lambda x: x[1])
#     > for the recommendation we sortin our data based on the similarity...then we will see our data are gets sort
#       but we lose the position of the index (suppose 1 movie have similarity with index numer 2,6,109,500,1009),
#       then we use 'enumerate'
#     > enumerate gives an object then we using list function = list(enumerate(similarity))
#     > after solve the index problem we use 'sorted' function for sorting
#     > 'reverse' = for the decsending order we use revesre
#     > after sorting we see, our sorted list is sort on the basis of index.....but we need on the basis of similarity
#     > so we use 'lambda' function = lambda function tells the program to sort our data on basis of second feature 
#       which is similarity 
#     > and the similarity is the "Cosine" distance
#       so we
#         sorted(list(enumerate(distance)),reverse =True , key = lambda x: x[1])


#      > if we wants check the index then we simply use i[0]
#      > for the name of the movie we use " new_df.iloc[i[0]].title "


# In[69]:


recommend('Avatar')


# In[70]:


recommend('Batman Begins')


# In[71]:


recommend('Spider-Man 3')


# In[72]:


recommend('Man of Steel')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




