import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random
import requests
from streamlit_lottie import st_lottie

# Load and preprocess data
@st.cache_data
def load_data():
    try:
        df_zomato = pd.read_csv("zomato.csv", encoding='ISO-8859-1')
    except UnicodeDecodeError:
        df_zomato = pd.read_csv("zomato.csv", encoding='latin1')
    
    df_zomato['Currency'] = df_zomato['Currency'].replace('Botswana Pula(P)', 'Philippine Peso (₱)')
    df_zomato['Restaurant_Details'] = df_zomato['City'] + ' ' + df_zomato['Cuisines'] + ' ' + df_zomato['Average Cost for two'].astype(str)
    df_zomato['Restaurant_Details'].fillna("", inplace=True)
    
    currency_mapping = {
        'Philippine Peso (₱)': 'PHP',
        'Brazilian Real(R$)': 'BRL',
        'Dollar($)': 'USD',
        'Emirati Diram(AED)': 'AED',
        'Indian Rupees(Rs.)': 'INR',
        'Indonesian Rupiah(IDR)': 'IDR',
        'NewZealand($)': 'NZD',
        'Pounds(å£)': 'GBP',
        'Qatari Rial(QR)': 'QAR',
        'Rand(R)': 'ZAR',
        'Sri Lankan Rupee(LKR)': 'LKR',
        'Turkish Lira(TL)': 'TRY'
    }
    
    df_zomato['currency_symbol'] = df_zomato['Currency'].map(currency_mapping)
    
    return df_zomato

# Create TF-IDF matrix and cosine similarity
@st.cache_resource
def create_similarity_matrix(df):
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(df['Restaurant_Details'])
    return cosine_similarity(tfidf_matrix, tfidf_matrix)

# Get restaurant recommendations
def get_recommendations(restaurant_name, cosine_sim, data, city, cuisine, budget_range, top_n=3):
    if restaurant_name == "All":
        # If "All" is selected, randomly choose restaurants that match the criteria
        filtered_data = data
        if city != "All":
            filtered_data = filtered_data[filtered_data['City'] == city]
        if cuisine != "All":
            filtered_data = filtered_data[filtered_data['Cuisines'].str.contains(cuisine, na=False)]
        filtered_data = filtered_data[
            (filtered_data['Average Cost for two'] >= budget_range[0]) & 
            (filtered_data['Average Cost for two'] <= budget_range[1])
        ]
        
        recommended_restaurants = filtered_data.sample(min(top_n, len(filtered_data))).to_dict('records')
    else:
        restaurant_index = data[data['Restaurant Name'] == restaurant_name].index[0]
        similarity_scores = list(enumerate(cosine_sim[restaurant_index]))
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        
        recommended_restaurants = []
        for index, score in similarity_scores[1:]:
            restaurant = data.iloc[index]
            if (city == "All" or restaurant['City'] == city) and \
               (cuisine == "All" or cuisine in restaurant['Cuisines']) and \
               (budget_range[0] <= restaurant['Average Cost for two'] <= budget_range[1]):
                recommended_restaurants.append({
                    'Restaurant Name': restaurant['Restaurant Name'],
                    'Cuisines': restaurant['Cuisines'],
                    'Average Cost for two': restaurant['Average Cost for two'],
                    'Currency': restaurant['Currency'],
                    'City': restaurant['City']
                })
            if len(recommended_restaurants) == top_n:
                break
    
    return recommended_restaurants

# Load Lottie animation
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Streamlit app
def main():
    st.title("Restaurant Recommender")
    
    # Load Lottie animation
    lottie_url = "https://lottie.host/d9eb34f1-86a0-4a10-a917-2427530db9ad/rXLRA001T4.json"  # URL to a food-related Lottie animation
    lottie_animation = load_lottieurl(lottie_url)
    
    # Sidebar with About section and Lottie animation
    st.sidebar.title("About")
    if lottie_animation:
        st_lottie(lottie_animation, speed=1, height=130, key="sidebar_animation")
    st.sidebar.info("This app recommends restaurants based on your preferences.")
    st.sidebar.info("Use the options to filter and find your perfect dining spot!")
    
    # Add LinkedIn link
    st.sidebar.markdown("### Connect with me")
    linkedin_url = "https://www.linkedin.com/in/naveen-mukala/"  # Replace with your actual LinkedIn URL
    st.sidebar.markdown(f"[![LinkedIn](https://content.linkedin.com/content/dam/me/business/en-us/amp/brand-site/v2/bg/LI-Bug.svg.original.svg)]({linkedin_url})")
    st.sidebar.markdown(f"[Visit my LinkedIn profile]({linkedin_url})")

    # Load data
    df_zomato = load_data()
    cosine_sim = create_similarity_matrix(df_zomato)
    
    # User input
    col1, col2 = st.columns(2)
    with col1:
        city = st.selectbox("Select a city:", ["All"] + sorted(df_zomato['City'].unique()))
    with col2:
        cuisines = df_zomato['Cuisines'].str.split(', ', expand=True).stack().unique()
        cuisine = st.selectbox("Select a cuisine category:", ["All"] + sorted(cuisines))
    
    # Dynamic budget range and currency based on selected city
    if city == "All":
        min_cost = int(df_zomato['Average Cost for two'].min())
        max_cost = int(df_zomato['Average Cost for two'].max())
        currency = "Mixed Currencies"
    else:
        city_data = df_zomato[df_zomato['City'] == city]
        min_cost = int(city_data['Average Cost for two'].min())
        max_cost = int(city_data['Average Cost for two'].max())
        currency = city_data['currency_symbol'].iloc[0]
    
    budget_range = st.slider(f"Average Cost for two in {currency}:", min_cost, max_cost, (min_cost, max_cost))
    
    # Filter restaurants based on selected city, cuisine, and budget
    df_filtered = df_zomato
    if city != "All":
        df_filtered = df_filtered[df_filtered['City'] == city]
    if cuisine != "All":
        df_filtered = df_filtered[df_filtered['Cuisines'].str.contains(cuisine, na=False)]
    df_filtered = df_filtered[
        (df_filtered['Average Cost for two'] >= budget_range[0]) & 
        (df_filtered['Average Cost for two'] <= budget_range[1])
    ]
    
    restaurant_name = st.selectbox("Select a restaurant:", ["All"] + sorted(df_filtered['Restaurant Name'].unique()))
    top_n = st.slider("Number of recommendations:", 1, 10, 3)
    
    if st.button("Get Recommendations"):
        recommendations = get_recommendations(restaurant_name, cosine_sim, df_zomato, city, cuisine, budget_range, top_n)
        
        st.subheader("Recommended Restaurants:")
        if recommendations:
            for i, resto in enumerate(recommendations, 1):
                st.write(f"{i}. {resto['Restaurant Name']}")
                st.write(f"   Cuisine: {resto['Cuisines']}")
                st.write(f"   Average Cost for two: {resto['Currency']} {resto['Average Cost for two']}")
                st.write(f"   City: {resto['City']}")
                st.write("---")
        else:
            st.write("No recommendations found based on the selected criteria.")

if __name__ == "__main__":
    main()