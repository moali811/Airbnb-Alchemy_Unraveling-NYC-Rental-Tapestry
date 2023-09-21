#!/usr/bin/env python
# coding: utf-8

# # Data Analysis of New York Rentals with Python

#  

# By Mohamad Ali  
# 01-07-2023

# 
# 

# ## Introduction

#  

# In this article, we will conduct a data exploratory analysis of the "New York City Airbnb" dataset using Python and various libraries such as Pandas, Matplotlib, and Seaborn.
# 
# The objective of this analysis is to gain insights into the distribution of rentals, price ranges, popular locations, and provide rental recommendations along with other tools to help you decide between the best and most affordable areas based on the Airbnb dataset for New York City.
# 
# Our analysis will cover several key aspects, such as:
# 
# 
# A. **Proportion of rentals by accommodation type:** We will create a pie chart to visually represent the proportion of rentals corresponding to each accommodation type, including entire homes/apartments, private rooms, and shared rooms.
# 
# B. **Distribution of rentals among the five boroughs of NY:** We will generate a bar plot to illustrate the distribution of rentals among the five boroughs of NY, namely Manhattan, Brooklyn, Queens, Bronx, and Staten Island.
# 
# C. **Price distribution and range:** We will use a histogram to analyze the price distribution of rentals in New York City and highlight the range of prices available for each accommodation type.
# 
# D. **Differentiation of prices by accommodation type:** To provide more insights into pricing trends, we will create a bar plot that distinguishes prices among the available accommodation types for each borough.
# 
# E. **Most popular locations to rent a lodge:** Using a bar plot, we will identify and visualize the most reviewed neighborhoods, helping us determine popular rental choices based on user reviews.
# 
# F. **Rental recommendations:** We will identify the most affordable rentals with the highest number of reviews for each accommodation type in a couple of neighborhoods within NY city for comparision.
# 
# G. **Final Map:** Visualizing the distribution of Airbnb apartments, their availability, and top venues in the reccommended neighborhood(s), to make informed decisions about where to stay or invest in real estate in New York.
# 
# 

# 
# 

# ## Data Preparation

# 
# 

# In[19]:


# Import necessary libraries

import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
import folium
from folium.plugins import MarkerCluster, HeatMap, MeasureControl
from IPython.display import display
import ipywidgets as widgets
from geopy.geocoders import Nominatim
import numpy as np
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import tkinter as tk
from tkinter import ttk
import webbrowser
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import colorlover as cl
from colorama import init, Fore, Style
from sklearn.cluster import KMeans
import requests
import seaborn as sns
# Call sns.set() to set the default Seaborn style
sns.set()
# SSL certificate verification
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


# ### Gathering Data

# In[3]:


# Referencing our data to it's original source(s) and importing the necessary libraries.

# Data source:
data_url1 = 'https://www.kaggle.com/dgomonov/new-york-city-airbnb-open-data'
data_url2 = 'http://insideairbnb.com/get-the-data.html'
data_url3 = 'https://raw.githubusercontent.com/Jotaherrer/DataAnalysis/master/new_york.csv'
data_CSV_file_url = 'https://1drv.ms/u/s!AvBIUg0IX58S5HM_gTKeNleITI-O?e=XmHljX'
data_url1_display = f'[Data Source 1]({data_url1})'
data_url2_display = f'[Data Source 2]({data_url2})'
data_url3_display = f'[Data Source 3]({data_url3})'
print(f'# Data source(s):\n{data_url1_display}\n{data_url2_display}\n{data_url3_display}')


# ### Loading Data

# In[4]:


# Load the dataset from a CSV file, utilizing the Pandas’s function .read_csv().
airbnb_data = pd.read_csv(data_url3)


# ### Exploring Data

# Let’s take a look at the basic information of the dataset, including features, data types, columns, entries and every value available at first sight, utilizing the .info() function:

# In[5]:


# Data basic info
print("\n# Basic information of Airbnb dataset:")
airbnb_data.info()


#  

# As it can be seen in the Gist, the dataset contains **16** columns, **48895** entries/rows, and different data types, such as Numpy integers, Objects, and Numpy floats along with some values and nulls.
# 
# Among the features or columns, we can find an ID, name of the landlord, rental ID, host name, borough, and other valuable information from which we’ll extract conclusions later on.

# ### Cleaning Data

# Let's proceed with reviewing duplicated values and replacing “missings” or "null" values from the dataset with zero, as nulls are mostly focused in the “number of reviews” and “last review” columns and they have no useful application in our analysis:
# 

# In[6]:


# Review duplicated values, it should return zero
duplicated_rows = airbnb_data.duplicated()
print("\n# Number of duplicated rows: ", duplicated_rows.sum())

# Replace missing/null values with zero
airbnb_data.fillna(0, inplace=True)


# Let's preview the first rows of our manipulated dataset and confirm its final shape, before we move on to the analysis:

# In[7]:

print("\n# Display the first few rows of the DataFrame:")
print(airbnb_data.head())


# In[8]:

print("\n# Shape of the data:")
print(airbnb_data.shape)


#  

#  ## Data Analysis & Visualization

# ### Descriptive Statistics

# Now that we’ve cleaned the dataset by removing unnecessary features, we can proceed with the analysis. To gain a better understanding of the distribution and characteristics of the Airbnb listings data, it is helpful to display a statistical summary. This summary inlcudes information for each numerical column in the dataset such as: counts, means, standard deviations, minimum and maximum values, as well as quartiles.

# In[9]:


print("\n# Statistical summary of the DataFrame:")
print(airbnb_data.describe())


#  

# Based on the statistical summary, we can derive *initial* insights and observations into the dataset:
# 
# 1. **Price Distribution:** The prices of listings vary significantly, ranging from 0 to 10,000. The average price is 152.72, but there is a high standard deviation of 240.15. This indicates a wide range of pricing, with some listings priced much higher than the average.
# 
# 2. **Minimum Stay Requirement:** Listings typically have a minimum stay requirement of around 7 nights on average. However, there is considerable variation among listings, as indicated by the standard deviation of 20.51.
# 
# 3. **Reviews and Ratings:** Listings receive an average of 23.27 reviews, suggesting a moderate level of engagement and feedback from guests. Hosts receive approximately one review per month on average (mean of 1.09), indicating ongoing guest activity.
# 
# 4. **Host Listings and Availability:** On average, hosts have around 7 listings (mean of calculated_host_listings_count). This suggests the presence of hosts with multiple listings, which can impact competition and availability. The average availability throughout the year is 112.78 days, indicating that listings are generally accessible for a significant portion of the year.
# 
# 5. **Outliers:** Certain columns, such as price and minimum_nights, exhibit maximum values that are notably higher than the 75th percentile. This suggests the presence of outliers, where some listings have exceptionally high prices or minimum stay requirements.

#  ### Visualizations

# In the next stages we will further analyse and visualize the data to provide a more comprehensive understanding of rental distributions, price ranges, popular locations, and provide rental recommendations based on the Airbnb data. To start with, let's address the following questions:

# #### A. What proportion of the rentals correspond to each accommodation type?

# In[10]:


# 1 - Pie chart

# Specify the colors for the pie chart slices
colors = ['darkcyan', 'steelblue', 'powderblue']

# Calculate the count of rentals for each accommodation type
rental_counts = airbnb_data['room_type'].value_counts()

# Create the pie trace
pie_trace = go.Pie(
    labels=rental_counts.index,
    values=rental_counts.values,
    textinfo='percent',
    marker=dict(colors=colors),
)

# Create layout
layout = go.Layout(
    title=dict(
        text=f'Rental Distribution by Accommodation Type (Total Rentals: {len(airbnb_data)})',
        x=0.5,  # Set the x position to the middle of the plot
        font=dict(size=15, color='navy'),
    ),
    plot_bgcolor='white',
    paper_bgcolor='white',
)

# Create the figure
fig = go.Figure(data=[pie_trace], layout=layout)

# Display the plot
fig.show()


#  

# Understanding the proportions of different accommodation types can help hosts and travelers alike in making informed decisions regarding their preferences, budget, and desired experience during their stay.
# 
# 
# Our analysis revealed that **52%** of Airbnb rentals are entire homes/apartments, while **46%** are private room rentals, with shared rooms comprising only **2%** of the dataset. This indicates a strong preference for private accommodations among guests, highlighting the importance of privacy and comfort.

# #### B. How are rentals distributed among the five boroughs of New York City?

# In[11]:

# 2 - Bar plot for rentals distribution by location

# Calculate the count of rentals for each borough
borough_counts = airbnb_data['neighbourhood_group'].value_counts().reset_index()
borough_counts.rename(columns={'index': 'borough', 'neighbourhood_group': 'count'}, inplace=True)

# Create a bar trace
bar_trace = go.Bar(
    x=borough_counts['borough'],
    y=borough_counts['count'],
    marker=dict(
        color=borough_counts['count'],
    )
)

# Create layout
layout = go.Layout(
    title='Rental Distribution by Location (Borough)',
    title_font=dict(size=15, color='navy'),
    title_x=0.5,
    xaxis=dict(title='Location', title_font=dict(size=12)),
    yaxis=dict(title='Number of Rentals', title_font=dict(size=12)),
    plot_bgcolor='white',
    paper_bgcolor='white',
)

# Create the figure
fig = go.Figure(data=[bar_trace], layout=layout)

# Display the plot
fig.show()




#  

#   
# The distribution of rentals across the five mentioned boroughs clearly demonstrates the overwhelming popularity of **Manhattan** and **Brooklyn** as the top choices for Airbnb listings. Together, these two boroughs account for a significant majority, surpassing 40,000 rentals. This pronounced concentration indicates a strong preference among visitors to New York for accommodations situated in these vibrant and dynamic parts of New York City.

# #### C. What’s the price distribution and what’s the range of fair prices available?

# In[12]:


# 3 - Histogram plot with price distribution

# Extract price data and calculate descriptive statistics
price_data = airbnb_data['price']
price_stats = price_data.describe()
q25, q75 = price_stats['25%'], price_stats['75%']
iqr_range = q75 - q25

# Create a histogram trace
histogram_trace = go.Histogram(x=price_data, histnorm='density', marker_color='lightblue', opacity=0.7)

# Create layout
layout = go.Layout(
    title='Price Distribution of Rentals in New York City',
    title_font=dict(size=15, color='navy'),
    title_x=0.5,
    xaxis=dict(title='Price (USD) per Night', title_font=dict(size=12)),
    yaxis=dict(title='Density', title_font=dict(size=12)),
    plot_bgcolor='white',
    bargap=0.1
)

# Create descriptive statistics text
descriptive_text = f"<b>Price Statistics:</b><br>"
descriptive_text += "<br>".join([f"{stat.capitalize()}: {price_stats[stat]:.2f}" for stat in ['mean', 'std', 'min', '25%', '50%', '75%', 'max']])
descriptive_text += f"<br>IQR Range: {q25:.2f} - {q75:.2f}"

# Create an annotation with descriptive statistics
descriptive_annotation = go.layout.Annotation(
    x=1,
    y=0.5,
    xref='paper',
    yref='paper',
    text=descriptive_text,
    showarrow=False,
    font=dict(family='Arial', size=10),
    align='left',
    bgcolor='whitesmoke',
    opacity=0.7
)

# Add the histogram trace and annotation to the figure
fig = go.Figure(data=[histogram_trace], layout=layout)
fig.update_layout(annotations=[descriptive_annotation])

# Set the theme to 'plotly_white' for better visibility
fig.update_layout(template='plotly_white')

# Display the plot
fig.show()


#  

# The price distribution of rentals in vibrant New York City exhibits a right-skewed pattern, indicating that the majority of listings are priced below USD 1,000, within the range of **USD 69-175** per night, forming a cluster of the most common prices. This sweet spot represents the interquartile range (IQR), capturing the middle 50% of prices. Meanwhile, the median price of **USD 106** indicates that half of the rentals are available below this value. On average, rentals in the city are priced at **USD 152.72**, reflecting the overall cost of accommodations.
# 
# However, it is worth noting that the market offers a diverse range of price options, spanning from lower-priced rentals to premium offerings. Some exceptional listings are priced as high as **USD 10,000**, representing the upper end of the price spectrum.

# #### D. How do prices vary among different accommodation types in New York City?

# In[13]:


# 4 - Bar plot with price to location distribution

# Group the Airbnb data by 'neighbourhood_group' and 'room_type', and calculate the mean of 'price'
loc_price = airbnb_data.groupby(['neighbourhood_group', 'room_type'])['price'].mean().reset_index()

# Define the colors for the bars
colors = ['darkcyan', 'steelblue', 'powderblue']

# Create a list to store the bar traces
bar_traces = []

# Iterate over each unique room type
for i, room_type in enumerate(loc_price['room_type'].unique()):
    # Filter the data for the current room type
    filtered_data = loc_price[loc_price['room_type'] == room_type]
    
    # Create a bar trace for the current room type
    bar_trace = go.Bar(
        x=filtered_data['neighbourhood_group'],
        y=filtered_data['price'],
        marker_color=colors[i],
        name=room_type,
        hovertemplate='Price: $%{y:.2f}<br>Borough: %{x}<extra></extra>'
    )
    
    # Add the bar trace to the list
    bar_traces.append(bar_trace)

# Create the bar plot
fig = go.Figure(data=bar_traces)

# Customize the plot layout
fig.update_layout(
    title='Average Price Distribution by Location and Accommodation Type',
    xaxis_title='Borough',
    yaxis_title='Price (USD) per Night',
    title_font=dict(size=15, color='navy'),
    title_x=0.5,   # Center the title horizontally
    plot_bgcolor='white',
    paper_bgcolor='white',
    font=dict(size=12),
    showlegend=True
)

# Display the plot
fig.show()


#  

# Stepping into the vibrant world of New York City's accommodations with the captivating bar plot above. It highlights the average prices for various accommodation types, unveils the mesmerizing tapestry of price distribution across the city's boroughs. As we journey through the visual symphony, it becomes apparent that certain boroughs have higher average prices across all accommodation types, while others offer more affordable options.
# 
# The plot unravels the crown jewel of **Manhattan**, boasting the highest average prices that reflect its allure and cosmopolitan charm. Meanwhile, **Brooklyn** and **Queens** shimmer with a diverse palette of prices, catering to a variety of tastes and budgets, with comparatively lower average prices for certain accommodation types.

# #### E. Which are the most popular locations to rent a lodge?

# In[14]:


# 5 - Most reviewed spots

# Sort the dataset based on the number of reviews in descending order
review_sorted = airbnb_data.sort_values('number_of_reviews', ascending=False)

# Select the top 50 reviewed spots along with their neighborhood, borough, and number of reviews
top_reviewed = review_sorted.loc[:, ['neighbourhood', 'neighbourhood_group', 'number_of_reviews']].head(50)

# Calculate the average number of reviews for each neighborhood and sort the data in descending order
top_reviewed_avg = top_reviewed.groupby(['neighbourhood', 'neighbourhood_group']).mean().reset_index().sort_values('number_of_reviews', ascending=False)

# Create a bar plot to visualize the average number of reviews for each neighborhood
fig = px.bar(top_reviewed_avg, x='number_of_reviews', y='neighbourhood', color='neighbourhood_group', orientation='h',
             labels={'number_of_reviews': 'Avg No. of Reviews', 'neighbourhood_group': 'Borough'},
             hover_data={'number_of_reviews': True, 'neighbourhood': False, 'neighbourhood_group': True},
             color_discrete_sequence=px.colors.qualitative.Set2)  # Use a color palette for distinct borough colors

# Customize the plot
fig.update_layout(
    title='Most Reviewed Rentals by Neighbourhood',
    title_font=dict(size=15, color='navy'),
    plot_bgcolor='white',
    paper_bgcolor='white',
    font=dict(size=12),
    legend_title='Borough',
    legend_y=0.5,  # Center the legend vertically
    yaxis=dict(title='Neighbourhood', categoryorder='total ascending'),
    xaxis=dict(title='Average Number of Reviews'),
    title_x=0.5,   # Center the title horizontally
)

# Display the plot
fig.show()



#  

# The graph highlights neighborhoods that have garnered a significant number of reviews, indicating their popularity among Airbnb guests. By comparing the review counts across color-coded bars representing different boroughs, we can identify the boroughs with higher review activity.
# 
# Pay attention to the lesser-known neighborhoods that have managed to accumulate a substantial number of reviews. These hidden gems often offer unique and memorable experiences, presenting an opportunity for travelers to discover off-the-beaten-path destinations.
# 
# Based on our analysis, it is evident that there is a high demand for accommodations in the boroughs of **Manhattan** and **Brooklyn**. This leads us to question the cost of staying in these locations:

# #### F. Is it feasible to find affordable yet pleasant rentals in New York City?

# Look no further! With just a few simple inputs, we can narrow down our rental search to the desired neighborhood. This enables us to identify the most affordable rentals among the highly reviewed options, ensuring that we find the most suitable place to stay.
# 
# For a vibrant and affordable experience with soul food, jazz clubs, and friendly neighbors, **Harlem** in Upper Manhattan is an excellent choice. It's diverse, welcoming, and popular among expats.
# 
# However, for a quieter lifestyle, endless green spaces, and yet, a budget-friendly option in Manhattan, the **Upper West Side (UWS)** is ideal. It offers great parks, family-friendly amenities, and excellent schools, making it attractive for raising kids.
# 
# Let's elevate the art of decision-making by comparing the results with a grouped bar chart for the most affordable prices between Harlem and Upper West Side (UWS), and through a captivating map, that seamlessly blends data and aesthetics. To discover the hidden gems and highly acclaimed accommodations that have delighted countless guests and explore the vibrant surroundings, from trendy eateries to iconic landmarks that define NYC's essence:

# In[123]:


# Foluim map with venues

# Function to filter and sort available rentals in a neighborhood based on number of reviews and price
def filter_and_sort_rentals(neighborhood, percentile):
    rentals = airbnb_data[(airbnb_data['neighbourhood'] == neighborhood) & 
                          (airbnb_data['number_of_reviews'] >= np.quantile(airbnb_data['number_of_reviews'], percentile)) &
                          (airbnb_data['availability_365'] > 0)].sort_values('price')
    return rentals

# Function to get the cheapest rentals for each accommodation type in a neighborhood
def get_cheapest_rentals(rentals):
    cheapest_rentals = {}
    accommodation_types = ['Entire home/apt', 'Private room', 'Shared room']
    
    for accommodation_type in accommodation_types:
        filtered_rentals = rentals[rentals['room_type'] == accommodation_type]
        if len(filtered_rentals) > 0:
            cheapest_rental = filtered_rentals.iloc[0]
            cheapest_rentals[accommodation_type] = cheapest_rental
    
    return cheapest_rentals

# 6 -  Function to plot the grouped bar chart
def plot_grouped_bar_chart(neighborhoods, cheapest_rentals):
    labels = ['Entire home/apt', 'Private room', 'Shared room']
    color_palette = px.colors.qualitative.Set2
    data = []
    
    for i, neighborhood in enumerate(neighborhoods):
        prices = [cheapest_rentals[neighborhood].get(accommodation_type, {}).get('price') for accommodation_type in labels]
        trace = go.Bar(
            x=labels,
            y=prices,
            name=neighborhood,
            text=['${}'.format(price) if price is not None else None for price in prices],
            hovertemplate=(
                '<b>Price:</b> %{text}<br>' +
                '<b>Availability:</b> %{customdata[15]} days<br>' +
                '<b>No. of Reviews:</b> %{customdata[11]}<br>' +
                '<b>Neighborhood:</b> %{customdata[5]}, %{customdata[4]}<br>' +
                '<b>ID:</b> %{customdata[0]}'
            ),
            customdata=[cheapest_rentals[neighborhood].get(accommodation_type, {}) for accommodation_type in labels],
            marker_color=color_palette[i % len(color_palette)]  # Cycle through the colors in the palette
        )
        data.append(trace)
        
    layout = go.Layout(
        title=f'Most Reviewed-Cheapest Rentals by Neighborhood and Accommodation Type',
        title_font=dict(size=15, color='navy'),
        xaxis=dict(title='Accommodation Type'),
        yaxis=dict(title='Price (USD) per Night'),
        title_x=0.5,
        barmode='group',
        plot_bgcolor='white',
        paper_bgcolor='white'
    )

    fig = go.Figure(data=data, layout=layout)
    fig.show()

# Function to get nearby venues using Foursquare API
def get_nearby_venues(lat, lon):
    # Foursquare API credentials
    CLIENT_ID = 'JQ5A2C1YB2GJUEZ0FGISIVQNYGLZ1GM2X4AAANLKF45WW5G2'
    CLIENT_SECRET = 'ELHXLFL0GKHNFUBEG3R0ADSYRNFQ4F5KTJJWQQQZO4TTYUHG'
    VERSION = '20230704'  # Use the current date as the version
    LIMIT = 5
    radius = 500

    url = f'https://api.foursquare.com/v2/venues/explore?client_id={CLIENT_ID}&client_secret={CLIENT_SECRET}&v={VERSION}&ll={lat},{lon}&radius={radius}&limit={LIMIT}'
    response = requests.get(url).json()

    venues = []
    if 'response' in response and 'groups' in response['response']:
        items = response['response']['groups'][0]['items']
        for item in items:
            venue = {
                'name': item['venue']['name'],
                'lat': item['venue']['location']['lat'],
                'lng': item['venue']['location']['lng']
            }
            categories = item['venue'].get('categories', [])
            if categories:
                venue['category'] = categories[0]['name']
            else:
                venue['category'] = ''
            venues.append(venue)

    return venues

# 7 - Function to generate the map of recommended rentals with accommodation type filter
def generate_rentals_map(rentals):
    # Generate the rentals map
    rentals_map = folium.Map(location=[rentals['latitude'].mean(), rentals['longitude'].mean()], zoom_start=12, control_scale=True)

    # Create a marker cluster group for venues
    venue_cluster = MarkerCluster().add_to(rentals_map)

    # Define icons or colors for each accommodation type
    icon_mapping = {
        'Entire home/apt': 'home',
        'Private room': 'bed',
        'Shared room': 'users'
    }

    # Define color ranges and corresponding colors
    color_mapping = {
        (0, 50): 'darkred',
        (50, 100): 'red',
        (100, 200): 'orange',
        (200, 350): 'pink',
        (350, float('inf')): 'purple'
    }

    # Add accommodation type filter legend
    accommodation_types = ['Entire home/apt', 'Private room', 'Shared room']

    legend_html = '''
        <div style="position: fixed;
                    top: 58%; left: 11px; transform: translateY(-50%);
                    width: 140px; border:1px solid grey; z-index:9999;
                    font-size:11px; background-color:white;
                    opacity:0.9; padding: 10px;">
        <hr style="margin: 5px 0;">

        <div style="text-align: center; margin-bottom: 10px;"><b>Lodge Type</b></div>
    '''
    for accommodation_type in accommodation_types:
        icon = icon_mapping.get(accommodation_type, 'home')
        color = 'black'
        legend_html += f'''
            <div style="display: flex; align-items: center; margin-bottom: 5px;">
                <div style="margin-right: 5px; color: {color};"><i class="fa fa-{icon} fa-lg"></i></div>
                <div style="flex: 1;">{accommodation_type}</div>
            </div>
        '''
        legend_html += '<hr style="margin: 5px 0;">'

    legend_html += '<div style="margin-top: 10px;"><hr style="margin: 5px 0;"></div>'

    legend_html += '''
        <div style="text-align: center; margin-bottom: 5px;"><b>Price Range</b></div>
    '''
    for price_range, color in color_mapping.items():
        legend_html += '''
            <div style="display: flex; align-items: center; margin-bottom: 5px;">
                <div style="margin-right: 5px;"><i class="fa fa-map-marker fa-lg" style="color: {};"></i></div>
                <div style="flex: 1;">${} - ${}</div>
            </div>
        '''.format(color, price_range[0], price_range[1])
        legend_html += f'<hr style="margin: 5px 0; border-top: 2px solid {color};">'

    legend_html += '''
        <div style="display: flex; align-items: center; margin-bottom: 5px;">
            <div style="margin-right: 5px; color: cadetblue;"><i class="fa fa-map-marker fa-lg"></i></div>
            <div style="flex: 1;">Lowest Price</div>
        </div>
        <hr style="margin: 5px 0;">
    '''
    
    legend_html += '<div style="margin-top: 10px;"><hr style="margin: 5px 0;"></div>'
    
    legend_html += '''
        <div style="text-align: center; margin-bottom: 5px;"><b>Venues</b></div>
    '''
    legend_html += '''
        <div style="display: flex; align-items: center; margin-bottom: 5px;">
            <div style="margin-right: 5px; color: blue;"><i class="fa fa-circle fa-lg"></i></div>
            <div style="flex: 1;">Venues</div>
        </div>
        <hr style="margin: 5px 0;">
    '''

    legend_html += '</div>'

    rentals_map.get_root().html.add_child(folium.Element(legend_html))

    # Find the lowest priced accommodation for each accommodation type
    lowest_price_rentals = rentals.groupby('room_type').apply(lambda x: x[x['price'] == x['price'].min()]).reset_index(drop=True)

    # Create a feature group for each accommodation type
    feature_groups = {accommodation_type: folium.FeatureGroup(name=accommodation_type) for accommodation_type in accommodation_types}

    # Iterate over the filtered accommodation data and add markers to the respective feature groups
    for _, rental in rentals.iterrows():
        accommodation_type = rental['room_type']
        price = rental['price']
        availability = rental['availability_365']
        
        # Skip rentals with 0 availability
        if availability == 0:
            continue

        icon = icon_mapping.get(accommodation_type, 'home')  # Use 'home' icon as default

        for price_range, color in color_mapping.items():
            if price_range[0] <= price < price_range[1]:
                marker_color = color
                break
        else:
            marker_color = 'gray'  # Default color if price doesn't match any range

        popup_html = f"<b>{rental['name']}</b><br>"
        popup_html += f"<b>Price:</b> ${rental['price']}<br>"
        popup_html += f"<b>Availability:</b> {rental['availability_365']}<br>"
        popup_html += f"<b>No. of Reviews:</b> {rental['number_of_reviews']}<br>"
        popup_html += f"<b>Neighborhood:</b> {rental['neighbourhood']}, {rental['neighbourhood_group']}<br>"
        popup_html += f"<b>ID:</b> {rental['id']}"

        # Check if the rental is the lowest priced for its accommodation type
        if rental['room_type'] in lowest_price_rentals['room_type'].values and rental['price'] == lowest_price_rentals.loc[lowest_price_rentals['room_type'] == rental['room_type'], 'price'].values[0]:
            folium.Marker(
                location=[rental['latitude'], rental['longitude']],
                popup=folium.Popup(popup_html, max_width=300),
                icon=folium.Icon(color='cadetblue', icon=icon, prefix='fa', prefix_color='white'),
                tooltip=f"{rental['room_type']} - ${rental['price']} (Lowest Price)"
            ).add_to(feature_groups[accommodation_type])
        else:
            folium.Marker(
                location=[rental['latitude'], rental['longitude']],
                popup=folium.Popup(popup_html, max_width=300),
                icon=folium.Icon(color=marker_color, icon=icon, prefix='fa'),
                tooltip=f"{rental['room_type']} - ${rental['price']}"
            ).add_to(feature_groups[accommodation_type])

        # Get nearby venues using your existing 'get_nearby_venues' function
        nearby_venues = get_nearby_venues(rental['latitude'], rental['longitude'])
        if nearby_venues:
            for venue in nearby_venues:
                folium.CircleMarker(
                    location=[venue['lat'], venue['lng']],
                    radius=7,
                    color='blue',
                    fill=False,
                    fill_color='blue',
                    fill_opacity=0.7,
                    tooltip=f"{venue['name']} - {venue['category']}"
                ).add_to(venue_cluster)

    # Add the feature groups to the map with layer control
    for accommodation_type, feature_group in feature_groups.items():
        rentals_map.add_child(feature_group)
    
    folium.LayerControl(position='topleft').add_to(rentals_map)
    
    # Display the rentals map (in Jupyter) and save it as a HTML file
    display(rentals_map)
    map_file_path = 'rentals_map.html'
    rentals_map.save(map_file_path)
    # Open the HTML file in Safari
    webbrowser.open(map_file_path, new=2)

# Prompt for user input
def prompt_user():
    neighborhoods = []
    percentiles = []

    while True:
        neighborhood = input("\nEnter the name of the neighborhood (or type '#' to quit): ")
        if neighborhood == '#':
            break

        while True:
            try:
                percentile = float(input("Enter the percentile for min number of reviews (0-100th): "))
                if 0 <= percentile <= 100:
                    break
                else:
                    print("Invalid percentile value. Please enter a value between 0 and 100.")
            except ValueError:
                print("Invalid input. Please enter a valid percentile value.")

        neighborhoods.append(neighborhood)
        percentiles.append(percentile / 100)  # Convert percentage to decimal

    # Plot the grouped bar chart for the final inputs
    rentals = []    # List to store filtered rentals for each neighborhood
    for neighborhood, percentile in zip(neighborhoods, percentiles):
        rental = filter_and_sort_rentals(neighborhood, percentile)
        rentals.append(rental)

    # Get the cheapest rentals for each neighborhood
    cheapest_rentals = {}
    for i, neighborhood in enumerate(neighborhoods):
        cheapest_rentals[neighborhood] = get_cheapest_rentals(rentals[i])

    # Display the cheapest rentals
    for neighborhood in neighborhoods:
        print(f"\n# Cheapest Rentals in {neighborhood}:")
        for accommodation_type, rental in cheapest_rentals[neighborhood].items():
            print(f"\n{accommodation_type} (Id:{rental['id']}, Reviews:{rental['number_of_reviews']}): ${rental['price']} per night")
        print()

    # Plot the grouped bar chart
    plot_grouped_bar_chart(neighborhoods, cheapest_rentals)

    # Generate the rentals map
    rentals_map = pd.concat(rentals)
    generate_rentals_map(rentals_map)

# Call the prompt_user function
prompt_user()

#  

#  

# The map shows the distribution of Airbnb apartments in Harlem and UWS. As we embarked on our journey through these neighborhoods in Manhattan, we discovered that the rental prices vary between Harlem and UWS. The cheapest Entire Home/Apt in Harlem is priced at **USD 49** per night, while in UWS, it is **USD 75** per night.
# 
# However, for those seeking an affordable yet charming experience, the Private Room option emerged as the true gem in both Harlem and UWS, with prices ranging from **USD 25** to **USD 30** per night.
# 
# The map also shows popular attractions, restaurants, or landmarks in these areas, through the blue "circular" markers representing the top venues in each neighborhood based on the Foursquare API data, reading the venue name, category.
# 
# *By visualizing the distribution of Airbnb apartments, their availability, and the top venues in the desired neighborhood(s), we can make informed decisions about where to stay or invest in real estate in these neighborhoods.*

#  

# # Conclusion: 

#   

# In this article, we have explored various aspects of the New York City rental market using Airbnb data:
# 
# We began by analyzing the **distribution of rentals** among the five boroughs of New York City. The bar plot visualization provided insights into the popularity of each borough, allowing us to make informed decisions about our preferred location.
# 
# Next, we delved into the **pricing patterns** across different accommodation types and neighborhoods. The bar plot with price-to-location distribution showcased the average prices for various room types in each borough, giving us an understanding of the cost implications associated with our accommodation choices.
# 
# Additionally, we examined the **most reviewed** spots, highlighting the neighborhoods that have received the highest number of reviews. This information can guide us in selecting popular locations that have been well-received by previous guests.
# 
# By providing these insights and visualizations, we hope to facilitate your decision-making process and enable you to optimize your lodging experience in New York City.
# 
# If you found this article helpful or have any thoughts to share, please feel free to reach out. Your feedback and engagement motivate us to continue sharing valuable information.
# 
# **Safe travels and happy renting!**

#  
