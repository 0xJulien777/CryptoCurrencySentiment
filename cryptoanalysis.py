import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer, TextClassificationPipeline
from praw import Reddit
import datetime
import time
import plotly.express as px
import matplotlib.dates as mdates
import seaborn as sns
import numpy as np
import plotly.graph_objects as go 

# Initialize CryptoBERT
model_name = "ElKulako/cryptobert"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
crypto_sentiment_pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer, max_length=64, truncation=True, padding='max_length')

# Reddit API credentials and initialization
reddit = Reddit(client_id='J19cj9blE8Eho_CFG7dOjQ',
                client_secret='-4WQFAt8ClCfngBd430Gy4snx2udIg',
                user_agent='YOUR_USER_AGENT')

# Function to fetch posts

def fetch_reddit_posts(subreddit, keyword, limit=15):
    posts = []
    subreddit_obj = reddit.subreddit(subreddit)
    if keyword:  # If there's a keyword, search for it
        submissions = subreddit_obj.search(keyword, limit=limit)
    else:  # If not, fetch the most recent posts
        submissions = subreddit_obj.new(limit=limit)

    for submission in submissions:
        # Convert the UTC timestamp to a datetime object
        created_time = datetime.datetime.fromtimestamp(submission.created_utc)
        posts.append({
            "title": submission.title,
            "text": submission.selftext,
            "date": created_time
        })
    return posts

# Function to analyze sentiment with CryptoBERT

def analyze_sentiment(posts):
    results = []
    for post in posts:
        # Combine title and text for sentiment analysis
        combined_text = post["title"] + ". " + post["text"]
        output = crypto_sentiment_pipeline(combined_text)
        for sentiment in output:
            # Append date information to each sentiment record
            sentiment['date'] = post['date']
        results.extend(output)
    return pd.DataFrame(results)


st.sidebar.title("Cryptocurrency Sentiment Analysis 🕵️ ")
page = st.sidebar.selectbox("Choose a page 👆", ["Home Page 🏠" ,"Search by Token 🔍✨", "Search by SubReddit 📂", "Global Trends 🌍💱", "Sentiment Analysis Over Time"])





if page == "Home Page 🏠":
    st.title("Welcome to Cryptocurrency Sentiment Analysis 🕵️")
    
    st.markdown("""
        ## What is Cryptocurrency Sentiment Analysis?
        
        In the rapidly evolving world of cryptocurrencies, staying informed about market sentiment is crucial. Our app leverages advanced Natural Language Processing (NLP) techniques, specifically using the CryptoBERT model, to analyze sentiments expressed in cryptocurrency-related discussions across various subreddits.
        
        ## How It Works
        
        - **Search by Token 🔍✨**: Enter a specific cryptocurrency token (e.g., $BTC, $ETH, $ADA) to see the sentiment analysis based on recent Reddit posts. This feature helps you gauge the current market sentiment towards a particular cryptocurrency.
        
        - **Search by SubReddit 📂**: Dive into specific subreddits to understand the sentiment of discussions. It’s a great way to see what the community feels about broader topics or specific events.
        
        - **Global Trends 🌍💱**: Get a bird's-eye view of the sentiment across the most talked-about cryptocurrencies on Reddit. This feature aggregates sentiment across multiple tokens, providing a comprehensive look at the market.
        
        ## Why Use Cryptocurrency Sentiment Analysis 🤔 ?
        
        - **Make Informed Decisions 📜**: By understanding market sentiment, you can make more informed decisions about your cryptocurrency investments.
        
        - **Stay Ahead 🏎️**: Sentiment analysis can sometimes give early signals about market movements, helping you stay a step ahead.
        
        - **Community Insights 💡**: Gain insights into the cryptocurrency community's thoughts and feelings, providing a deeper understanding beyond just price movements.
        
        ## Contribute and Collaborate
        
        We're on a mission to make this tool even more powerful with additional features like Twitter API integration. Contributions are welcome! Here’s our 🔺 Avax 🔺address for support: `0xe5d5D0078E3a5afE22727F9c16dc545Ae3CA0046`. Together, we can make this happen! 🤝⚒️
        
        ## 🏁 Get Started 🏁
        
        Explore our features by choosing a page from the sidebar and dive into the world of cryptocurrency sentiment analysis. Let's explore what the community is feeling today! 🚀
    """)
    
    st.image("https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExeXV6eDZvMW1ibHB6a3I3ODB4ZHNieXFkcTBnZGFkYjViY3JrcHFkZyZlcD12MV9naWZzX3NlYXJjaCZjdD1n/l0HlDDyxBfSaPpU88/giphy.gif", use_column_width=True)



if page == 'Search by Token 🔍✨':
    st.write("Let's build a more powerful tool with Twitter API, Together we can make it happen! 🤝⚒️. 🔺Avax 🔺 adress for contributions : 0xe5d5D0078E3a5afE22727F9c16dc545Ae3CA0046 ")
    st.title('Search by Crypto Asset 🚀 👨‍🚀')
    st.write("This feature allows you to analyze the sentiment of discussions related to a specific cryptocurrency. Enter the token symbol and get insights about the market sentiment ! 🕵️")
    crypto = st.text_input("Enter the cryptocurrency💲(e.g., $BTC, $ETH, $AVAX):", value="$AVAX")
    if st.button("Analyze Sentiment 🕵️"):
        
        with st.spinner("🌐 Fetching Reddit posts and analyzing sentiment..."):
            posts = fetch_reddit_posts("CryptoCurrency", crypto, 10)

            bar= st.progress(33)
            time.sleep(1)
            
            
            if posts:
                sentiment_df = analyze_sentiment(posts)
                # Map model output to readable sentiments
                label_map = {'LABEL_0': 'Bullish', 'LABEL_1': 'Neutral', 'LABEL_2': 'Bearish'}
                sentiment_df['label'] = sentiment_df['label'].map(lambda x: label_map.get(x, x))
                
                # Calculate and display sentiment probabilities
                sentiment_probs = sentiment_df['label'].value_counts(normalize=True) * 100
                sentiment_colors = {'Bullish': 'green', 'Neutral': '#FAA300', 'Bearish': 'red'}

                bullish_prob = sentiment_probs.get('Bullish', 0)
                bearish_prob = sentiment_probs.get('Bearish', 0)
                neutral_prob = sentiment_probs.get('Neutral', 0)
                # Use a horizontal bar plot
                fig, ax = plt.subplots(facecolor='#0B0E12')
                sentiment_probs.plot(kind='barh', color=[sentiment_colors[label] for label in sentiment_probs.index], ax=ax)

                # Set the properties of the axes
                ax.set_facecolor('#0B0E12')
                ax.tick_params(colors='white', which='both')
                ax.spines['bottom'].set_color('black')
                ax.spines['left'].set_color('black')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

                bar.progress(67)

                # Set labels and title with white color for visibility on dark background
                ax.set_xlabel('Proportion (%)', color='white')
                ax.set_ylabel(' ', color='black')
                ax.set_title('Sentiment Analysis Proportion', color='white')

                # Change the color of all tick labels to white
                for label in ax.get_xticklabels() + ax.get_yticklabels():
                    label.set_color('white')

                plt.tight_layout()
                st.pyplot(fig)


                if bullish_prob > bearish_prob:
                    st.image("https://media.giphy.com/media/ri3uvuWKPc9UWFMKXq/giphy.gif?cid=790b76112owvypuvaxz5dmganztdnbh3a1kpbysejhes2k61&ep=v1_gifs_search&rid=giphy.gif&ct=g", caption="Market sentiment is Bullish! 🚀")
                else :
                    st.image("https://media.giphy.com/media/ZX1knKISQDLZS/giphy.gif?cid=790b76115e76f0nozau8c40s0i26w7fb5styxkc02afn2gpd&ep=v1_gifs_search&rid=giphy.gif&ct=g", caption="Market sentiment is Bearish! 📉")

                bar.progress(100)
            else:
                st.write("No posts found for the given cryptocurrency ⛔ ")





                
elif page == 'Global Trends 🌍💱':
    st.title("Analyze Global Cryptocurrency Market Sentiment 🌍💱")
    st.write("🔺Avax 🔺 adress for contributions : 0xe5d5D0078E3a5afE22727F9c16dc545Ae3CA0046")
    st.write("Let's build a more powerful tool with Twitter API, Together we can make it happen! 🤝⚒️ ")
    st.write("Click Analyze button to get Insights about Global Cryptocurrency Market Sentiment 🌍💱")
    if st.button("Analyze Global Sentiment 🕵️"):
        with st.spinner("🌐 Fetching latest posts and analyzing sentiment..."):

            # Define a list of cryptocurrencies for global analysis
            cryptocurrencies = ['$BTC', '$ETH', '$AVAX', '$SOL', '$MATIC']
            
            # Initialize an empty DataFrame to store sentiment results
            global_sentiment_df = pd.DataFrame()

            progress_increment = 100 // len(cryptocurrencies)  
            current_progress = 0
            bar = st.progress(current_progress)
            
            for crypto in cryptocurrencies:
                posts = fetch_reddit_posts("CryptoCurrency", crypto, 10)
                if posts:
                    sentiment_df = analyze_sentiment(posts)
                    sentiment_df['crypto'] = crypto  # Add a column for the cryptocurrency
                    # Concatenate the results to the global dataframe
                    global_sentiment_df = pd.concat([global_sentiment_df, sentiment_df], ignore_index=True)
                    global_sentiment_normalized = global_sentiment_df.groupby(['crypto', 'label']).agg({'score': 'mean'}).groupby(level=0).apply(lambda x: x / x.sum()).reset_index()


                current_progress += progress_increment
                bar.progress(min(current_progress, 100))  # Update progress

            if not global_sentiment_df.empty:
                # Map model output to readable sentiments
                label_map = {'LABEL_0': 'Bullish', 'LABEL_1': 'Neutral', 'LABEL_2': 'Bearish'}
                global_sentiment_df['label'] = global_sentiment_df['label'].map(lambda x: label_map.get(x, x))
                global_sentiment_normalized = global_sentiment_df.groupby(['crypto', 'label']).agg({'score': 'mean'}).groupby(level=0).apply(lambda x: x / x.sum()).reset_index()
                # Calculate sentiment probabilities for each cryptocurrency
                global_sentiment_df['count'] = 1  # Add a column to assist in counting
                sentiment_summary = global_sentiment_df.groupby(['crypto', 'label']).count()['count'].unstack(fill_value=0)
                sentiment_summary_percentage = sentiment_summary.divide(sentiment_summary.sum(axis=1), axis=0) * 100


                st.title("Stacked bars plot of sentiment analysis distribution for each cryptocurrency 📊")

                sns.set_theme(style="darkgrid")
                fig, ax = plt.subplots(figsize=(10, 6))
                sentiment_summary_percentage.plot(kind='bar', stacked=True, color=['red', '#FAA300', 'green'], ax=ax)
                plt.title('Global Cryptocurrency Sentiment')
                plt.ylabel('Percentage')
                plt.xlabel('Cryptocurrency')
                plt.legend(title='Sentiment')
                st.pyplot(fig)



                st.title("Heatmap of Normalized Global Cryptocurrency Sentiment 🔥🚒")



                heatmap_data = pd.pivot_table(global_sentiment_normalized, values='score', index='crypto', columns='label').fillna(0)
                
                # Generate heatmap
                plt.figure(figsize=(10, 6))
                sns.heatmap(heatmap_data, annot=True, cmap='Greens', fmt=".2f")
                plt.title('Global Cryptocurrency Sentiment Heatmap')
                plt.ylabel('Cryptocurrency')
                plt.xlabel('Sentiment')
   
                st.pyplot(plt)


                sns.set_style("whitegrid")

                # Create the violin plot
                plt.figure(figsize=(10, 6))  # Set the figure size
                violin = sns.violinplot(x='label', y='score', data=sentiment_df, 
                                        inner='quartile',  # Show the quartiles
                                        palette='muted')  # Use a muted palette for a sophisticated look

                # Customize the visualization
                violin.set_title('Distribution of Sentiment Scores by Category', fontsize=15)
                violin.set_xlabel('Sentiment Category', fontsize=12)
                violin.set_ylabel('Sentiment Score', fontsize=12)

                # Improve legibility
                sns.despine(left=True)  # Remove the left spine for a cleaner look
                plt.tight_layout()

                # Display the plot in Streamlit
                st.pyplot()
            else:
                st.write("No posts found for the given cryptocurrency ⛔ ")

            bar.progress(100)  # Complete the progress

# Note: The rest of your Streamlit code remains unchanged.

        
    
 
elif page == "Search by SubReddit 📂":
    st.title("Analyze CryptoCurrency Sentiment by SubReddit 📂")
    st.write("🔺Avax 🔺 adress for contributions : 0xe5d5D0078E3a5afE22727F9c16dc545Ae3CA0046")
    st.write("Let's build a more powerful tool with Twitter API, Together we can make it happen! 🤝⚒️ ")

    # User inputs for subreddit selection and optional keyword filtering
    subreddit_input = st.text_input("🔍🌐 Enter the subreddit name (e.g., CryptoCurrency):", value="CryptoCurrency")
    keyword_input = st.text_input("🔑 Enter a keyword to filter (optional):", value="")
    limit_posts = st.number_input("Number of posts to analyze:", min_value=5, max_value=50, value=15, step=5)
    
    if st.button("Analyze Sentiment 🕵️"):
        with st.spinner("🌐 Fetching latest posts and analyzing sentiment..."):
            # Initialize progress
            bar = st.progress(0)

            # Fetch posts using the existing function with the user's input
            # If keyword is not provided (empty string), it fetches the latest posts regardless of keywords
            posts = fetch_reddit_posts(subreddit=subreddit_input, keyword=keyword_input, limit=limit_posts)
            bar.progress(33)  # Update progress after fetching posts

            if posts:
                time.sleep(1)  # Simulate processing time
                sentiment_results = analyze_sentiment(posts)
                bar.progress(67)  # Update progress after analyzing sentiment

                # Map model output to readable sentiments
                label_map = {'LABEL_0': 'Bullish', 'LABEL_1': 'Neutral', 'LABEL_2': 'Bearish'}
                sentiment_results['label'] = sentiment_results['label'].map(lambda x: label_map.get(x, x))
                
                # Calculate and display sentiment probabilities
                sentiment_probs = sentiment_results['label'].value_counts(normalize=True) * 100
                sentiment_colors = {'Bullish': 'green', 'Neutral': '#FAA300', 'Bearish': 'red'}

                st.write(sentiment_probs)

                # Use a horizontal bar plot
                fig, ax = plt.subplots(facecolor='#0B0E12')
                sentiment_probs.plot(kind='barh', color=[sentiment_colors[label] for label in sentiment_probs.index], ax=ax)

                # Set the properties of the axes
                ax.set_facecolor('#0B0E12')
                ax.tick_params(colors='white', which='both')
                ax.spines['bottom'].set_color('black')
                ax.spines['left'].set_color('black')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

                # Set labels and title with white color for visibility on dark background
                ax.set_xlabel('Proportion (%)', color='white')
                ax.set_ylabel(' ', color='black')
                ax.set_title('Sentiment Analysis Proportion', color='white')

                # Change the color of all tick labels to white
                for label in ax.get_xticklabels() + ax.get_yticklabels():
                    label.set_color('white')

                plt.tight_layout()
                st.pyplot(fig)

                bar.progress(100)  # Complete the progress
            else:
                st.write(" ⚠️ ERROR : No posts were fetched, please check the subreddit name or try a different keyword. 🔐")



    # Reference
    st.markdown("Reference: ElKulako (2022). CryptoBERT: A Pre-trained NLP Model for Cryptocurrency Sentiment Analysis. IEEE Explore. Available at: [IEEE Xplore](https://ieeexplore.ieee.org/document/10223689)")


elif page == "Sentiment Analysis Over Time":
    st.write("🔺Avax 🔺 adress for contributions : 0xe5d5D0078E3a5afE22727F9c16dc545Ae3CA0046")
    st.title("Sentiment Analysis Over Time 📈")

    subreddit_input = st.text_input("Enter the subreddit name:", value="CryptoCurrency")
    keyword_input = st.text_input("Enter a keyword to filter (optional):", value="")
    limit_posts = st.number_input("Number of posts to analyze:", min_value=5, max_value=50, value=20, step=5)

    if st.button("Fetch and Analyze Posts"):
        with st.spinner("Fetching Reddit posts and analyzing sentiment..."):
            posts = fetch_reddit_posts(subreddit=subreddit_input, keyword=keyword_input, limit=limit_posts)
            if posts:
                sentiment_df = analyze_sentiment(posts)

                # Ensure 'date' column is datetime type for plotting
                sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
                sentiment_df.sort_values(by='date', inplace=True)

                # Group by both date and hour for finer granularity
                time_series_df = sentiment_df.groupby([sentiment_df['date'].dt.floor('H'), 'label']).size().unstack(fill_value=0)

                # Create the plot with fig, ax
                fig, ax = plt.subplots(figsize=(10, 6))
                time_series_df.plot(kind='line', ax=ax)

                # Set the labels and title
                ax.set_ylabel('Count')
                ax.set_title('Sentiment Analysis Over Time')

                # Set x-axis major locator to hourly and formatter to show hour:minute
                ax.xaxis.set_major_locator(mdates.HourLocator())
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

                # Ensure labels are rotated for readability
                ax.tick_params(axis='x', rotation=45)

                # Set the legend title
                ax.legend(title='Sentiment')

                # Apply a tight layout
                plt.tight_layout()

                # Display the plot in Streamlit
                st.pyplot(fig)
    st.markdown("Reference: ElKulako (2022). CryptoBERT: A Pre-trained NLP Model for Cryptocurrency Sentiment Analysis. IEEE Explore. Available at: [IEEE Xplore](https://ieeexplore.ieee.org/document/10223689)")
