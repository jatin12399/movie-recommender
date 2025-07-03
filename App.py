import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process
import requests


@st.cache_data
def load_data():
    df = pd.read_csv("imdb_dataset.csv")
    df = df[['movie_title', 'genres', 'director_name', 'actor_1_name', 'actor_2_name', 'actor_3_name', 'imdb_score',
             'movie_imdb_link']]
    df.dropna(subset=['movie_title'], inplace=True)

    for col in ['genres', 'director_name', 'actor_1_name', 'actor_2_name', 'actor_3_name']:
        df[col] = df[col].fillna('')

    df["combined_features"] = df.apply(lambda
                                           row: f"{row['genres']} {row['director_name']} {row['actor_1_name']} {row['actor_2_name']} {row['actor_3_name']}",
                                       axis=1)

    cv = CountVectorizer(stop_words='english')
    count_matrix = cv.fit_transform(df["combined_features"])
    cosine_sim = cosine_similarity(count_matrix)

    df = df.reset_index()
    df['clean_title'] = df['movie_title'].str.strip().str.lower()
    return df, cosine_sim


df, cosine_sim = load_data()


# Poster from OMDb API
def fetch_movie_details(title):
    try:
        url = f"http://www.omdbapi.com/?t={title.strip()}&apikey=140ff9bb"
        res = requests.get(url).json()
        return {
            "poster": res.get("Poster", ""),
            "actors": res.get("Actors", ""),
            "rating": res.get("imdbRating", "N/A"),
            "year": res.get("Year", ""),
            "genre": res.get("Genre", ""),
            "plot": res.get("Plot", "")
        }
    except:
        return {
            "poster": "",
            "actors": "",
            "rating": "N/A",
            "year": "",
            "genre": "",
            "plot": ""
        }


# Movie Recommender
def get_matches(query):
    titles = df['movie_title'].dropna().unique().tolist()
    matches = process.extract(query, titles, limit=10)
    return [match[0] for match in matches if match[1] > 60]


def recommend_movie(title, n=5):
    title_clean = title.strip().lower()
    matched = process.extractOne(title_clean, df['clean_title'].tolist())
    if not matched or matched[1] < 60:
        return None, []

    idx = df[df['clean_title'] == matched[0]].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n + 1]
    movie_indices = [i[0] for i in sim_scores]
    return df.loc[idx], df.loc[movie_indices]


# Streamlit UI
st.set_page_config(page_title="Movie Recommender", layout="wide")

st.markdown("""
    <style>
    .stApp {
        background-image: url("https://wallpaperaccess.com/full/1567673.jpg");
        background-size: cover;
        background-repeat: no-repeat;
        color: white;
    }
    .block-container {
        background-color: rgba(0, 0, 0, 0.6);
        padding: 2rem;
        border-radius: 15px;
    }
    .title {
        color: #fff;
        text-shadow: 1px 1px 5px #000;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="block-container">', unsafe_allow_html=True)
st.markdown("<h1 class='title'>🎬 Movie Recommender System 🎬</h1>", unsafe_allow_html=True)

query = st.text_input("🔍 Start typing a movie title:")
suggestions = get_matches(query) if query else []
selected_movie = st.selectbox("🎦 Select a movie:", suggestions) if suggestions else None

if selected_movie and st.button("🎥 Recommend"):
    selected_info, recommendations = recommend_movie(selected_movie)

    if selected_info is not None:
        movie_details = fetch_movie_details(selected_info['movie_title'])
        st.subheader(f"🎞️ {selected_info['movie_title'].strip()} ({movie_details['year']})")
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(movie_details['poster'] or "https://via.placeholder.com/300x450.png?text=No+Poster", width=250)
        with col2:
            st.markdown(f"**🎭 Genre:** {movie_details['genre'] or selected_info['genres']}")
            st.markdown(f"**🎬 Director:** {selected_info['director_name']}")
            st.markdown(
                f"**🧑‍🤝‍🧑 Actors:** {movie_details['actors'] or ', '.join([selected_info['actor_1_name'], selected_info['actor_2_name'], selected_info['actor_3_name']])}")
            st.markdown(f"**⭐ IMDB Rating:** {movie_details['rating'] or selected_info['imdb_score']}")
            st.markdown(f"**🔗 IMDB Link:** [Click Here]({selected_info['movie_imdb_link']})")
            st.markdown(f"**📝 Plot:** {movie_details['plot']}")

        st.markdown("---")
        st.markdown("## 🔮 Recommended Movies")
        for i, row in recommendations.iterrows():
            rec_details = fetch_movie_details(row['movie_title'])
            with st.container():
                rec1, rec2 = st.columns([1, 3])
                with rec1:
                    st.image(rec_details['poster'] or "https://via.placeholder.com/150", width=150)
                with rec2:
                    st.markdown(f"### {row['movie_title'].strip()} ({rec_details['year']})")
                    st.markdown(f"**🎭 Genre:** {rec_details['genre']}")
                    st.markdown(f"**🧑‍🤝‍🧑 Actors:** {rec_details['actors']}")
                    st.markdown(f"**⭐ IMDB Rating:** {rec_details['rating']}")
                    st.markdown(f"[🔗 View on IMDB]({row['movie_imdb_link']})")
                    st.markdown(f"**📝 Plot:** {rec_details['plot']}")
                    st.markdown("---")

    else:
        st.warning("Movie not found. Try another title.")
else:
    st.info("Start typing to search a movie name.")

st.markdown("</div>", unsafe_allow_html=True)
