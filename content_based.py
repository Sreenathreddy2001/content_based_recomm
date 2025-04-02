{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 359,
   "id": "0dcbba77-c495-439a-b58d-3358bda610f3",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer\n",
    "from sklearn.metrics.pairwise import pairwise_distances, linear_kernel\n",
    "import ast\n",
    "from gensim.models import Word2Vec\n",
    "from nltk.tokenize import word_tokenize\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 360,
   "id": "f7659069-0f50-4552-b6b9-82154c7d2898",
   "metadata": {},
   "outputs": [],
   "source": [
    "links=pd.read_csv(\"links.csv\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 361,
   "id": "bcd1790b-d952-47c1-b7c7-1873ec57b8a8",
   "metadata": {},
   "outputs": [],
   "source": [
    "tags=pd.read_csv(\"tags.csv\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 362,
   "id": "9f544704-c58b-462d-9850-be56c79432cb",
   "metadata": {},
   "outputs": [],
   "source": [
    "ratings=pd.read_csv(\"ratings.csv\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 363,
   "id": "1be7b595-f617-4ffa-b384-07d72d2ec03c",
   "metadata": {},
   "outputs": [],
   "source": [
    "movies.drop_duplicates(inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 364,
   "id": "b476f66a-ae12-480d-87e2-78b145edfeb8",
   "metadata": {},
   "outputs": [],
   "source": [
    "movies=pd.read_csv(\"movies.csv\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 365,
   "id": "f117a20f-6c5b-49ba-af66-78b1a0bcfcaf",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>movieId</th>\n",
       "      <th>title</th>\n",
       "      <th>genres</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>Toy Story (1995)</td>\n",
       "      <td>Adventure|Animation|Children|Comedy|Fantasy</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>Jumanji (1995)</td>\n",
       "      <td>Adventure|Children|Fantasy</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>Grumpier Old Men (1995)</td>\n",
       "      <td>Comedy|Romance</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>Waiting to Exhale (1995)</td>\n",
       "      <td>Comedy|Drama|Romance</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>Father of the Bride Part II (1995)</td>\n",
       "      <td>Comedy</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9737</th>\n",
       "      <td>193581</td>\n",
       "      <td>Black Butler: Book of the Atlantic (2017)</td>\n",
       "      <td>Action|Animation|Comedy|Fantasy</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9738</th>\n",
       "      <td>193583</td>\n",
       "      <td>No Game No Life: Zero (2017)</td>\n",
       "      <td>Animation|Comedy|Fantasy</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9739</th>\n",
       "      <td>193585</td>\n",
       "      <td>Flint (2017)</td>\n",
       "      <td>Drama</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9740</th>\n",
       "      <td>193587</td>\n",
       "      <td>Bungo Stray Dogs: Dead Apple (2018)</td>\n",
       "      <td>Action|Animation</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9741</th>\n",
       "      <td>193609</td>\n",
       "      <td>Andrew Dice Clay: Dice Rules (1991)</td>\n",
       "      <td>Comedy</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>9742 rows × 3 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "      movieId                                      title  \\\n",
       "0           1                           Toy Story (1995)   \n",
       "1           2                             Jumanji (1995)   \n",
       "2           3                    Grumpier Old Men (1995)   \n",
       "3           4                   Waiting to Exhale (1995)   \n",
       "4           5         Father of the Bride Part II (1995)   \n",
       "...       ...                                        ...   \n",
       "9737   193581  Black Butler: Book of the Atlantic (2017)   \n",
       "9738   193583               No Game No Life: Zero (2017)   \n",
       "9739   193585                               Flint (2017)   \n",
       "9740   193587        Bungo Stray Dogs: Dead Apple (2018)   \n",
       "9741   193609        Andrew Dice Clay: Dice Rules (1991)   \n",
       "\n",
       "                                           genres  \n",
       "0     Adventure|Animation|Children|Comedy|Fantasy  \n",
       "1                      Adventure|Children|Fantasy  \n",
       "2                                  Comedy|Romance  \n",
       "3                            Comedy|Drama|Romance  \n",
       "4                                          Comedy  \n",
       "...                                           ...  \n",
       "9737              Action|Animation|Comedy|Fantasy  \n",
       "9738                     Animation|Comedy|Fantasy  \n",
       "9739                                        Drama  \n",
       "9740                             Action|Animation  \n",
       "9741                                       Comedy  \n",
       "\n",
       "[9742 rows x 3 columns]"
      ]
     },
     "execution_count": 365,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "movies"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 366,
   "id": "0be0a832-c141-4efe-8087-2012d6d8dc01",
   "metadata": {},
   "outputs": [],
   "source": [
    "title_final=[]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 367,
   "id": "bc8456ac-f55c-4722-897f-347db02d49ba",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "for i in movies.title:\n",
    "    title_final.append((i.split('(')[0]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 368,
   "id": "15a7301c-f297-44f7-bc7c-9e1342d06771",
   "metadata": {},
   "outputs": [],
   "source": [
    "movies['title']=title_final"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 369,
   "id": "bce692cd-423f-40dd-8122-064424b47f8e",
   "metadata": {},
   "outputs": [],
   "source": [
    "movies['genres']=movies.genres.str.replace('|',',')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 370,
   "id": "204922ec-7340-420e-8aaa-8fbabce1dde9",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>movieId</th>\n",
       "      <th>title</th>\n",
       "      <th>genres</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>Toy Story</td>\n",
       "      <td>Adventure,Animation,Children,Comedy,Fantasy</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>Jumanji</td>\n",
       "      <td>Adventure,Children,Fantasy</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>Grumpier Old Men</td>\n",
       "      <td>Comedy,Romance</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>Waiting to Exhale</td>\n",
       "      <td>Comedy,Drama,Romance</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>Father of the Bride Part II</td>\n",
       "      <td>Comedy</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9737</th>\n",
       "      <td>193581</td>\n",
       "      <td>Black Butler: Book of the Atlantic</td>\n",
       "      <td>Action,Animation,Comedy,Fantasy</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9738</th>\n",
       "      <td>193583</td>\n",
       "      <td>No Game No Life: Zero</td>\n",
       "      <td>Animation,Comedy,Fantasy</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9739</th>\n",
       "      <td>193585</td>\n",
       "      <td>Flint</td>\n",
       "      <td>Drama</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9740</th>\n",
       "      <td>193587</td>\n",
       "      <td>Bungo Stray Dogs: Dead Apple</td>\n",
       "      <td>Action,Animation</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9741</th>\n",
       "      <td>193609</td>\n",
       "      <td>Andrew Dice Clay: Dice Rules</td>\n",
       "      <td>Comedy</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>9742 rows × 3 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "      movieId                                title  \\\n",
       "0           1                           Toy Story    \n",
       "1           2                             Jumanji    \n",
       "2           3                    Grumpier Old Men    \n",
       "3           4                   Waiting to Exhale    \n",
       "4           5         Father of the Bride Part II    \n",
       "...       ...                                  ...   \n",
       "9737   193581  Black Butler: Book of the Atlantic    \n",
       "9738   193583               No Game No Life: Zero    \n",
       "9739   193585                               Flint    \n",
       "9740   193587        Bungo Stray Dogs: Dead Apple    \n",
       "9741   193609        Andrew Dice Clay: Dice Rules    \n",
       "\n",
       "                                           genres  \n",
       "0     Adventure,Animation,Children,Comedy,Fantasy  \n",
       "1                      Adventure,Children,Fantasy  \n",
       "2                                  Comedy,Romance  \n",
       "3                            Comedy,Drama,Romance  \n",
       "4                                          Comedy  \n",
       "...                                           ...  \n",
       "9737              Action,Animation,Comedy,Fantasy  \n",
       "9738                     Animation,Comedy,Fantasy  \n",
       "9739                                        Drama  \n",
       "9740                             Action,Animation  \n",
       "9741                                       Comedy  \n",
       "\n",
       "[9742 rows x 3 columns]"
      ]
     },
     "execution_count": 370,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "movies"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 371,
   "id": "c3f9222c-ad72-41c2-b5ae-a22d9f0f5e6d",
   "metadata": {},
   "outputs": [],
   "source": [
    "genres=movies['genres'].tolist()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 372,
   "id": "26ae1c0b-92f3-4c02-a678-c4eb7c7dcadc",
   "metadata": {},
   "outputs": [],
   "source": [
    "tokenized_sentences = [word_tokenize(genre.lower()) for genre in genres]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 373,
   "id": "2093785d-7806-4403-bdcb-1d61e3a0da3e",
   "metadata": {},
   "outputs": [],
   "source": [
    "cleaned_tokenized_sentences = [[word for word in tokens if word != ','] for tokens in tokenized_sentences]\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 374,
   "id": "d3e79df1-4d1c-48b7-b935-40a6bfe8e53f",
   "metadata": {},
   "outputs": [],
   "source": [
    "model = Word2Vec(sentences=cleaned_tokenized_sentences, vector_size=100, window=5, min_count=1, sg=0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 375,
   "id": "0fc2be47-4623-45b1-8c1a-edb897fe6069",
   "metadata": {},
   "outputs": [],
   "source": [
    "def get_similar_genres(genres, top_n=5):\n",
    "    genre_vectors = [model.wv[genre] for genre in genres if genre in model.wv]\n",
    "    if not genre_vectors:\n",
    "        return []\n",
    "\n",
    "    avg_vector = sum(genre_vectors) / len(genre_vectors)\n",
    "    similar_genres = model.wv.similar_by_vector(avg_vector, topn=top_n)\n",
    "    return similar_genres\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 376,
   "id": "d4bd4fbe-faea-4159-83ea-d5e483f6ec50",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[('no', 0.9998588562011719),\n",
       " ('listed', 0.999525249004364),\n",
       " ('genres', 0.9995158910751343),\n",
       " ('(', 0.9994536638259888),\n",
       " (')', 0.9994218349456787)]"
      ]
     },
     "execution_count": 376,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "get_similar_genres(['drama','no'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 377,
   "id": "72ad999f-8651-4c22-aaa9-6346f5102627",
   "metadata": {},
   "outputs": [],
   "source": [
    "def recommend_movies_by_genre(user_genres, top_n=10):\n",
    "    similar_genres = get_similar_genres(user_genres)\n",
    "    recommended_movies = []\n",
    "    for genre, _ in similar_genres:   \n",
    "        count=0\n",
    "        while count < len(cleaned_tokenized_sentences):\n",
    "            if genre in cleaned_tokenized_sentences[count]:\n",
    "                recommended_movies.append(count)\n",
    "            count += 1\n",
    "        \n",
    "    return recommended_movies[0:top_n]\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 378,
   "id": "70aff88e-eea9-4b09-a199-150a6e1557da",
   "metadata": {},
   "outputs": [],
   "source": [
    "def recommend_movies(name_of_movie, top_n): \n",
    "    if name_of_movie == '':\n",
    "        return str(\"Please enter the movie\")\n",
    "    genre_of_movie = str(movies[movies['title'] == name_of_movie]['genres'].iloc[0]).split(',')\n",
    "    genre_of_movie = [element.lower() for element in genre_of_movie]\n",
    "    list1 = recommend_movies_by_genre(genre_of_movie, top_n)\n",
    "\n",
    "    \n",
    "    related_movies = []\n",
    "    for i in list1:\n",
    "        new_row = movies.iloc[i:i+1].to_dict()\n",
    "        related_movies.append(new_row)\n",
    "\n",
    "    formatted_related_movies = []\n",
    "    for j in related_movies:\n",
    "        new_row = {key: values[next(iter(values))] for key, values in j.items()}\n",
    "        formatted_related_movies.append(new_row)\n",
    "    return formatted_related_movies\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 379,
   "id": "1741307c-492b-455e-b65d-add7c6b5001f",
   "metadata": {},
   "outputs": [],
   "source": [
    "def recommend_movies(name_of_movie, top_n):\n",
    "    if not name_of_movie.strip():\n",
    "        return \"Please enter a valid movie name.\"\n",
    "    \n",
    "    # Normalize title casing and remove spaces\n",
    "    movies['title'] = movies['title'].str.strip().str.lower()\n",
    "    name_of_movie = name_of_movie.strip().lower()\n",
    "\n",
    "    # Filter movies by title\n",
    "    filtered_movies = movies[movies['title'] == name_of_movie]\n",
    "\n",
    "    # Check if movie exists in the dataset\n",
    "    if filtered_movies.empty:\n",
    "        return f\"Movie '{name_of_movie}' not found in dataset.\"\n",
    "\n",
    "    # Extract genres safely\n",
    "    genre_of_movie = filtered_movies['genres'].iloc[0]  # No IndexError now\n",
    "    genre_list = genre_of_movie.lower().replace(\" \", \"\").split(',')\n",
    "\n",
    "    # Get recommended movie indices\n",
    "    recommended_indices = recommend_movies_by_genre(genre_list, top_n)\n",
    "\n",
    "    # Convert indices to movie names\n",
    "    recommended_movie_names = movies.loc[recommended_indices, 'title'].tolist()\n",
    "\n",
    "    return recommended_movie_names\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 380,
   "id": "a96f56dd-9155-459f-9d05-eebee0f24c28",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['waiting to exhale',\n",
       " 'american president, the',\n",
       " 'nixon',\n",
       " 'casino',\n",
       " 'sense and sensibility']"
      ]
     },
     "execution_count": 380,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "recommend_movies('Flint',5)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
