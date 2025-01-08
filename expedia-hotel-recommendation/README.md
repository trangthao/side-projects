## Overview
This project focuses on predicting which hotel group (cluster) a user is most likely to book based on their search behavior and attributes. Expedia groups similar hotels into 100 clusters based on features like price, location, and customer ratings. The goal is to leverage customer data and provide personalized hotel recommendations to improve user satisfaction and platform engagement.

---

## Project Objectives
1. Predict the **hotel cluster** for a user based on search parameters and behavior.
2. Perform **exploratory data analysis (EDA)** to uncover trends and insights.
3. Build and evaluate machine learning models to optimize predictions.
I do not include the modularizing part for future scalability and ease of use here. 
---

## Dataset Description
The dataset includes various fields describing user search behavior and hotel attributes:

- **Date and Location:**
  - `date_time`: Timestamp of the search.
  - `srch_ci`, `srch_co`: Check-in and check-out dates.
  - `posa_continent`, `user_location_country`, `user_location_region`: User's location.
  - `hotel_continent`, `hotel_country`, `hotel_market`: Hotel details.
  
- **Search Details:**
  - `srch_adults_cnt`, `srch_children_cnt`: Number of adults and children in the search.
  - `srch_rm_cnt`: Number of hotel rooms requested.
  - `orig_destination_distance`: Distance between the user and the destination.

- **Behavioral Features:**
  - `is_mobile`: Whether the search was done on a mobile device.
  - `is_package`: Whether the search included a package (e.g., flight + hotel).

- **Target:**
  - `hotel_cluster`: ID of the hotel group to be predicted.

---

## Approach

### 1. Exploratory Data Analysis (EDA)
- Analyze the dataset to uncover trends and correlations.
- Identify missing values, outliers, and duplicate entries.
- Visualize patterns, such as the most popular booking times and user preferences.

### 2. Data Preprocessing
- Handle missing values through appropriate imputation techniques.
- Extract and engineer features such as:
  - Stay duration (`srch_co` - `srch_ci`).
  - Prior booking days (`srch_ci` - `date_time`).
  - Check-in day, month, and year.
- Normalize and scale data where necessary.

### 3. Model Building and Evaluation
- Train multiple classification models, including:
  - Random Forest
  - Logistic Regression
  - Gaussian Naive Bayes
  - K-Nearest Neighbors (KNN)
  - Decision Tree
  - XGBoost
- Compare models based on metrics like accuracy, precision, recall, and F1-score.

### 4. Solution Optimization
- Use hyperparameter tuning and feature selection to improve model performance.
- Save trained models as `.pkl` files for reuse.

---

## Visual Insights
- **Stay Duration Analysis:** Trends in the length of hotel stays (short vs. long trips).
- **Booking Behavior:** Patterns in booking times (e.g., weekdays vs. weekends).
- **Geographic Insights:** Most popular hotel clusters and user locations.

---

## Key Takeaways
### 1. **Geographic Dominance:**
   - The majority of bookings come from users in specific continents (e.g., Continent 3). This suggests that Expedia has a strong market presence in certain regions. 
   - Insight: Focus marketing and promotions on high-performing regions while creating targeted campaigns to engage users from underperforming continents.

### 2. **Duration of Stay Trends:**
   - Longer stays are associated with specific types of users or destinations, while short stays are common for business or quick leisure trips.
   - Insight: Categorize hotels into "Short Trips" and "Long Trips" to provide personalized recommendations and improve search functionality for users.

### 3. **Seasonal and Temporal Patterns:**
   - There are clear peaks in bookings during certain months (e.g., summer and holiday seasons) and days of the week (e.g., Fridays for weekend getaways).
   - Insight: Use seasonal trends to launch promotions, increase inventory for peak periods, and optimize pricing strategies.
