import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk

#Download NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

#Load CSV
df = pd.read_csv('Net.csv')

df.fillna('NA', inplace=True)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

#preprocess text by removing stopwords and lemmatizing
def preprocess_text(text):
    if not isinstance(text, str):
        text = 'NA'
    words = word_tokenize(text.lower())
    words = [lemmatizer.lemmatize(word) for word in words if word.isalnum() and word not in stop_words]
    return ' '.join(words)

# Preprocessing for the dataframe
def preprocess(row):
    text_elements = [
        row['Title'], row['Genre'], row['Tags'], row['Languages'], 
        row['Director'], row['Writer'], row['Actors'], row['Summary']
    ]
    processed_elements = [preprocess_text(str(element)) for element in text_elements]
    return ' '.join(processed_elements)

#Apply preprocessing
df['Processed_Text'] = df.apply(preprocess, axis=1)

#recommend movies based on user prompt
def recommend_movies(user_prompt, dataframe, top_n=10):
    user_prompt = preprocess_text(user_prompt)
    
    # Vectorize movie descriptions
    vectorizer = TfidfVectorizer()
    movie_vectors = vectorizer.fit_transform(dataframe['Processed_Text'])

    # Vectorize the user prompt
    user_vector = vectorizer.transform([user_prompt])

    # Calculate cosine similarity between user prompt and movie descriptions
    similarities = cosine_similarity(user_vector, movie_vectors)[0]

    # Rank and recommend movies
    dataframe['Similarity'] = similarities
    recommendations = dataframe.sort_values(by='Similarity', ascending=False).head(top_n)

    return recommendations[['Title', 'Genre', 'Languages', 'Director', 'Writer', 'Actors', 'Summary', 'Poster', 'Similarity', 'IMDb Score', 'Netflix Link']]

# Mood detection using TextBlob sentiment analysis
def detect_mood(user_prompt):
    blob = TextBlob(user_prompt)
    sentiment_score = blob.sentiment.polarity

    if sentiment_score > 0:
        return 'positive'
    elif sentiment_score < 0:
        return 'negative'
    else:
        return 'neutral'

#recommend movies based on user's mood and tags/genres
def recommend_movies_by_mood(user_prompt, dataframe, top_n=5):
    user_prompt = preprocess_text(user_prompt)
    
    # Detect mood from user prompt
    mood = detect_mood(user_prompt)
    
    # Mood-Tag Mapping
    mood_to_genre = {
        'positive': ['Thriller', 'Horror', 'Mystery', 'Adventure'],
        'negative': ['Comedy', 'Romance', 'Drama', 'Family'],
        'neutral': ['Comedy', 'Romance', 'Drama', 'Thriller', 'Mystery', 'Adventure', 'Family', 'Sci-Fi', 'Animation','Horror', 'Crime', 'Documentary']
    }
    
    # Filter movies based on mood and tags/genres
    filtered_df = dataframe[
        dataframe['Genre'].apply(lambda x: any(genre in x for genre in mood_to_genre.get(mood, []))) |
        dataframe['Tags'].apply(lambda x: any(tag in x for tag in mood_to_genre.get(mood, [])))
    ]

    if filtered_df.empty:
        filtered_df = dataframe  # If no movies match the mood and tags/genres, fall back to the full dataset

    # Vectorize the filtered movie descriptions (only 'Tags', 'Summary', 'Genre')
    filtered_df['Mood_Processed_Text'] = filtered_df.apply(lambda row: ' '.join([row['Tags'], row['Summary'], row['Genre']]), axis=1)
    vectorizer = TfidfVectorizer()
    movie_vectors = vectorizer.fit_transform(filtered_df['Mood_Processed_Text'])

    # Vectorize the user prompt
    user_vector = vectorizer.transform([user_prompt])

    # Calculate cosine similarity between user prompt and movie descriptions
    similarities = cosine_similarity(user_vector, movie_vectors)[0]

    # Rank and recommend movies
    filtered_df['Similarity'] = similarities
    recommendations = filtered_df.sort_values(by='Similarity', ascending=False).head(top_n)

    return recommendations[['Title', 'Genre', 'Languages', 'Director', 'Writer', 'Actors', 'Summary', 'Poster', 'Similarity', 'IMDb Score', 'Netflix Link']]

# Set page configuration
st.set_page_config(
    page_title="MovieMetrix",
    page_icon="production.png",  # You can use an emoji or a link to a favicon image
)

#CSS styling
st.markdown(
    """
    <style>
    body {
        background-image: url("https://wallpapers.com/images/hd/netflix-background-gs7hjuwvv2g0e9fj.jpg");
        color: #FFFFFF; /* White text color */
        font-family: 'Arial', sans-serif; /* Custom font */
    }

    .stApp {
        background: rgba(0, 0, 0, 0.7); /* Dark semi-transparent overlay */
        border-radius: 15px;
        padding: 20px;
    }

    .title {
        font-size: 3em;
        text-align: center;
        margin-bottom: 20px;
        font-weight: bold;
        color: #FF0000; /* Gold color */
    }

    .prompt-label {
        font-size: 1.2em;
        margin-bottom: 10px;
    }

    .section-title {
        font-size: 2em;
        margin-top: 30px;
        margin-bottom: 20px;
        text-align: center;
        color: #FFD700; /* Gold color */
    }

    .recommendation {
        border: 1px solid #FFD700; /* Gold border */
        padding: 20px;
        margin-bottom: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s, box-shadow 0.2s;
        background-color: rgba(0, 0, 0, 0.8);
    }

    .recommendation:hover {
        transform: translateY(-10px);
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.5);
    }

    .movie-metadata {
        font-size: 1.1em; /* Larger font size */
        margin-bottom: 5px;
    }

    .movie-metadata p {
        margin: 5px 0;
    }

    .recommendation img {
        width: 150px; /* Increase image size */
        height: auto;
        border-radius: 10px; /* Optional: Add border radius */
        float: left;
        margin-right: 20px;
    }

    a {
        color: #FFD700; /* Gold color for links */
        text-decoration: none;
    }

    a:hover {
        text-decoration: underline;
    }

    .stSidebar .element-container {
        background: rgba(0, 0, 0, 0.7); /* Dark semi-transparent background for sidebar */
        border-radius: 15px;
        padding: 15px;
    }

    .stSidebar .stButton button {
        background-color: #FFD700; /* Gold background for buttons */
        color: #000; /* Black text color for buttons */
        border: none;
        border-radius: 10px;
        padding: 10px;
        transition: background-color 0.2s, transform 0.2s;
    }

    .stSidebar .stButton button:hover {
        background-color: #FFAA00; /* Darker gold on hover */
        transform: scale(1.05);
    }

    .stTextInput textarea {
        background: rgba(255, 255, 255, 0.1); /* Slightly transparent white background */
        color: #FFFFFF; /* White text color */
        border: 1px solid #FFD700; /* Gold border */
        border-radius: 10px;
        padding: 10px;
    }

    .stTextInput textarea:focus {
        outline: none;
        border-color: #FFAA00; /* Darker gold on focus */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Define Streamlit app
def main():
    st.sidebar.title("Navigation")
    option = st.sidebar.selectbox("Select a page", ["Personalized Recommendations", "Mood-Based Recommendations", "Filter Movies"])

    st.markdown('<h1 class="title">MovieMetrix</h1>', unsafe_allow_html=True)

    if option == "Personalized Recommendations":
        st.markdown('<p class="prompt-label">Enter your preferences below:</p>', unsafe_allow_html=True)

        user_prompt = st.text_area("Type your preferences here:")
        if st.button("Get Recommendations"):
            if user_prompt.strip() == "":
                st.error("Please enter your preferences.")
            else:
                recommendations = recommend_movies(user_prompt, df)
                st.markdown('<h2 class="section-title">Top Recommendations</h2>', unsafe_allow_html=True)
                for index, row in recommendations.iterrows():
                    st.markdown(
                        f"""
                        <div class="recommendation">
                            <img src="{row['Poster']}" alt="Movie Poster">
                            <div class="movie-metadata">
                                <p><b>Title:</b> {row['Title']}</p>
                                <p><b>Genre:</b> {row['Genre']}</p>
                                <p><b>Languages:</b> {row['Languages']}</p>
                                <p><b>Director:</b> {row['Director']}</p>
                                <p><b>Writer:</b> {row['Writer']}</p>
                                <p><b>Actors:</b> {row['Actors']}</p>
                                <p><b>Summary:</b> {row['Summary']}</p>
                                <p><b>IMDb Score:</b> {row['IMDb Score']}</p>
                                <p><b>Similarity:</b> {row['Similarity']}</p>
                                <a href="{row['Netflix Link']}" target="_blank">Watch on Netflix</a>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

    elif option == "Mood-Based Recommendations":
        st.markdown('<p class="prompt-label">Enter your current mood below:</p>', unsafe_allow_html=True)

        user_prompt = st.text_area("How are you feeling today?")
        if st.button("Get Recommendations"):
            if user_prompt.strip() == "":
                st.error("Please enter your mood.")
            else:
                recommendations = recommend_movies_by_mood(user_prompt, df)
                st.markdown('<h2 class="section-title">Top Recommendations</h2>', unsafe_allow_html=True)
                for index, row in recommendations.iterrows():
                    st.markdown(
                        f"""
                        <div class="recommendation">
                            <img src="{row['Poster']}" alt="Movie Poster">
                            <div class="movie-metadata">
                                <p><b>Title:</b> {row['Title']}</p>
                                <p><b>Genre:</b> {row['Genre']}</p>
                                <p><b>Languages:</b> {row['Languages']}</p>
                                <p><b>Director:</b> {row['Director']}</p>
                                <p><b>Writer:</b> {row['Writer']}</p>
                                <p><b>Actors:</b> {row['Actors']}</p>
                                <p><b>Summary:</b> {row['Summary']}</p>
                                <p><b>IMDb Score:</b> {row['IMDb Score']}</p>
                                <p><b>Similarity:</b> {row['Similarity']}</p>
                                <a href="{row['Netflix Link']}" target="_blank">Watch on Netflix</a>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

    elif option == "Filter Movies":
        st.subheader("Filter Movies")
        filter_criteria = st.selectbox("Filter by", ["Director", "Actor", "Genre"])
        if filter_criteria == "Director":
            directors = df['Director'].unique()
            selected_director = st.selectbox("Select Director", directors)
            filtered_movies = df[df['Director'] == selected_director]
        elif filter_criteria == "Actor":
            actors = df['Actors'].str.split(',').explode().unique()
            selected_actor = st.selectbox("Select Actor", actors)
            filtered_movies = df[df['Actors'].str.contains(selected_actor)]
        elif filter_criteria == "Genre":
            genres = df['Genre'].str.split(',').explode().unique()
            selected_genre = st.selectbox("Select Genre", genres)
            filtered_movies = df[df['Genre'].str.contains(selected_genre)]

        for _, row in filtered_movies.iterrows():
            st.markdown(
                f"""
                <div class="recommendation">
                    <img src="{row['Poster']}" alt="Movie Poster">
                    <div class="movie-metadata">
                        <p><b>Title:</b> {row['Title']}</p>
                        <p><b>Genre:</b> {row['Genre']}</p>
                        <p><b>Languages:</b> {row['Languages']}</p>
                        <p><b>Director:</b> {row['Director']}</p>
                        <p><b>Writer:</b> {row['Writer']}</p>
                        <p><b>Actors:</b> {row['Actors']}</p>
                        <p><b>Summary:</b> {row['Summary']}</p>
                        <p><b>IMDb Score:</b> {row['IMDb Score']}</p>
                        <a href="{row['Netflix Link']}" target="_blank">Watch on Netflix</a>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

if __name__ == '__main__':
    main()
