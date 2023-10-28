# # Data Analysis of New York Rentals with Python

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
# Data source:
data_url1 = 'https://www.kaggle.com/dgomonov/new-york-city-airbnb-open-data'
data_url2 = 'http://insideairbnb.com/get-the-data.html'
data_url3 = 'https://raw.githubusercontent.com/moali811/Data-Science/main/airbnb_new_york.csv'
data_CSV_file_url = 'https://1drv.ms/u/s!AvBIUg0IX58S5HM_gTKeNleITI-O?e=XmHljX'
data_url1_display = f'[Data Source 1]({data_url1})'
data_url2_display = f'[Data Source 2]({data_url2})'
data_url3_display = f'[Data Source 3]({data_url3})'
print(f'# Data source(s):\n{data_url1_display}\n{data_url2_display}\n{data_url3_display}')

# Load the dataset from a CSV file, utilizing the Pandas’s function .read_csv().
airbnb_data = pd.read_csv(data_url3)


# ### Exploring Data

# Data basic info
print("\n# Basic information of Airbnb dataset:")
airbnb_data.info()

# Review duplicated values, it should return zero
duplicated_rows = airbnb_data.duplicated()
print("\n# Number of duplicated rows: ", duplicated_rows.sum())

# Replace missing/null values with zero
airbnb_data.fillna(0, inplace=True)

# Preview the first rows of our manipulated dataset and confirm its final shape, before we move on to the analysis:

print("\n# Display the first few rows of the DataFrame:")
print(airbnb_data.head())

print("\n# Shape of the data:")
print(airbnb_data.shape)

#  ## Data Analysis & Visualization

# ### Descriptive Statistics

print("\n# Statistical summary of the DataFrame:")
print(airbnb_data.describe())

#  ### Visualizations

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



# Based on our analysis, it is evident that there is a high demand for accommodations in the boroughs of **Manhattan** and **Brooklyn**. This leads us to question the cost of staying in these locations:

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
    
    # Display the rentals map (for Jupyter) and save it as a HTML file
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
