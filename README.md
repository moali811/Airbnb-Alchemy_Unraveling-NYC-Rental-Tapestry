# Data Analysis of New York Rentals with Python

By Mohamad Ali
01-07-2023

## Introduction
This project conducts a detailed analysis of New York City Airbnb listings using Python and various data analysis libraries. The dataset provides comprehensive information about rentals in New York City.

The **objective** of this analysis is to gain insights into the distribution of rentals, price ranges, popular locations, and provide rental recommendations along with other tools to help you decide between the best and most affordable areas based on the Airbnb dataset for New York City.

#### Our analysis will cover several key aspects, such as:

**A. Proportion of rentals by accommodation type:** We will create a pie chart to visually represent the proportion of rentals corresponding to each accommodation type, including entire homes/apartments, private rooms, and shared rooms.

**B. Distribution of rentals among the five boroughs of NY:** We will generate a bar plot to illustrate the distribution of rentals among the five boroughs of NY, namely Manhattan, Brooklyn, Queens, Bronx, and Staten Island.

**C. Price distribution and range:** We will use a histogram to analyze the price distribution of rentals in New York City and highlight the range of prices available for each accommodation type.

**D. Differentiation of prices by accommodation type:** To provide more insights into pricing trends, we will create a bar plot that distinguishes prices among the available accommodation types for each borough.

**E. Most popular locations to rent a lodge:** Using a bar plot, we will identify and visualize the most reviewed neighborhoods, helping us determine popular rental choices based on user reviews.

**F. Rental recommendations:** We will identify the most affordable rentals with the highest number of reviews for each accommodation type in a couple of neighborhoods within NY city for comparision.

**G. Final Map:** Visualizing the distribution of Airbnb apartments, their availability, and top venues in the reccommended neighborhood(s), to make informed decisions about where to stay or invest in real estate in New York.

## Data Gathering
The dataset was collected from multiple sources, including:
Kaggle: https://www.kaggle.com/dgomonov/new-york-city-airbnb-open-data
Inside Airbnb: http://insideairbnb.com/get-the-data.html
The data was then imported into a Pandas DataFrame for analysis: https://raw.githubusercontent.com/moali811/Data-Science/main/airbnb_new_york.csv

## Exploratory Data Analysis
- **Basic Information:** The dataset contains essential features such as accommodation type, price, location, and number of reviews. No duplicated rows were found.
  -

- **Descriptive Statistics:** A statistical summary provided insights into data central tendencies and spread.

  - **Price Distribution:** The prices of listings vary significantly, ranging from 0 to 10,000. The average price is 152.72, but there is a high standard deviation of 240.15. This indicates a wide range of pricing, with some listings priced much higher than the average.
  - **Minimum Stay Requirement:** Listings typically have a minimum stay requirement of around 7 nights on average. However, there is considerable variation among listings, as indicated by the standard deviation of 20.51.
  - **Reviews and Ratings:** Listings receive an average of 23.27 reviews, suggesting a moderate level of engagement and feedback from guests. Hosts receive approximately one review per month on average (mean of 1.09), indicating ongoing guest activity.
  - **Host Listings and Availability:** On average, hosts have around 7 listings (mean of calculated_host_listings_count). This suggests the presence of hosts with multiple listings, which can impact competition and availability. The average availability throughout the year is 112.78 days, indicating that listings are generally accessible for a significant portion of the year.
  - **Outliers:** Certain columns, such as price and minimum_nights, exhibit maximum values that are notably higher than the 75th percentile. This suggests the presence of outliers, where some listings have exceptionally high prices or minimum stay requirements.

## Data Visualization
1. **Accommodation Type Distribution:**
- A pie chart illustrates the distribution of rentals based on accommodation types, with entire homes/apartments being the majority (62.73%).
  ![newplot-2](https://github.com/moali811/Airbnb-Data-Analysis/assets/59733199/ee930302-2f97-4401-b084-93d135d7825c)

2. **Rental Distribution by Location (Borough):**
- A bar plot visualizes the number of rentals in each borough, highlighting Manhattan and Brooklyn as popular locations.
  ![newplot-3](https://github.com/moali811/Airbnb-Data-Analysis/assets/59733199/181604fa-64db-4d78-bac7-e978070215c5)

3. **Price Distribution:**
- A histogram shows the price distribution of rentals, indicating a right-skewed distribution with an interquartile range of $88 - $220.
  ![newplot-4](https://github.com/moali811/Airbnb-Data-Analysis/assets/59733199/1ab41cfa-d536-4757-8567-985e97fd3db7)

4. **Average Price by Location and Accommodation Type:**
- A grouped bar chart displays average rental prices for different accommodation types across boroughs, revealing price disparities.
  ![newplot-5](https://github.com/moali811/Airbnb-Data-Analysis/assets/59733199/fe9e58c9-077e-4772-95bc-06831a2dc22c)

5. **Most Reviewed Rentals:**
- A bar plot presents the top 50 most reviewed rentals by neighborhood, identifying areas with high rental demand.
  ![newplot-6](https://github.com/moali811/Airbnb-Data-Analysis/assets/59733199/7c1d8d9c-589c-4830-96c3-e456108fd346)

## Recommendations and Insights
- **High Demand Areas:**
  - Manhattan and Brooklyn are high-demand areas for rentals, suggesting lucrative opportunities for hosts.
  
- **Pricing Strategy:**
  - Hosts can optimize pricing based on accommodation type and location. Entire homes/apartments generally command higher prices.
 
- **Exploring Cheapest Rentals:**
  - By filtering rentals based on user-defined criteria (neighborhood and minimum number of reviews), the report identified the cheapest accommodations in specified neighborhoods, providing valuable insights for budget-conscious travelers.
  ![newplot-7](https://github.com/moali811/Airbnb-Data-Analysis/assets/59733199/2d7e2141-9f86-41c9-8e41-f15419255691)

- **Interactive Map:**
  - An interactive Folium map allows users to explore recommended rentals by accommodation type, price range, and nearby venues.
  <img width="1522" alt="Rentals_Map" src="https://github.com/moali811/Airbnb-Data-Analysis/assets/59733199/f8c01613-868c-4f73-84fd-9292922934fc">

## Conclusion
This analysis provides valuable insights for both hosts and travelers, aiding in strategic decision-making. Hosts can optimize their listings, while travelers can make informed choices. The interactive map enhances user experience, enabling seamless exploration of rental options.

*Note:* This analysis serves as a comprehensive overview based on the provided dataset. Further analysis can be conducted with additional data and specific business objectives in mind.

**Safe travels and happy renting!**

