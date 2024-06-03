# MovieMetrix
A Movie Recommendation System powered by NLP based on User-Prompt and Emotion or State of Mind
MovieMetrix is a personalized movie recommendation system built using Streamlit. The application leverages natural language processing (NLP) techniques and machine learning to provide tailored movie recommendations based on user preferences and moods.

Features
Personalized Recommendations: Input your movie preferences and get a list of recommended movies based on cosine similarity.
Mood-Based Recommendations: Enter your current mood to receive movie recommendations that match your sentiment using TextBlob sentiment analysis.
Filter Movies: Filter movies by director, actor, or genre for more targeted recommendations.
Installation
Clone the repository:
git clone https://github.com/Vishalmahajan1521/MovieMetrix.git
Navigate to the project directory:
cd MovieMetrix
Install the required packages:
pip install -r requirements.txt
Download the necessary NLTK data:
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
Usage
Ensure the Netflix Dataset.csv file containing movie data is in the project directory.
Run the Streamlit app:
streamlit run app.py
Use the sidebar to navigate between personalized recommendations, mood-based recommendations, and filtering options.
Enter your preferences or mood and get instant movie recommendations!
Project Structure
app.py: The main application file.
Netflix Dataset.csv: The dataset containing movie information.
requirements.txt: A file listing the required Python packages.
Technologies Used
Streamlit for building the web application
Pandas for data manipulation
scikit-learn for vectorization and similarity calculations
TextBlob for sentiment analysis
NLTK for text preprocessing
Screenshots
Screenshots of the application can be viewed here.
