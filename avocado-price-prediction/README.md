# Forecasting Avocado Prices (ARIMA vs Prophet)

## Problem Statement
ABC Farms, a Mexico-based company producing Hass avocados sold in the US, is analyzing sales and price trends to plan expansion.

## Data Description
The dataset represents weekly retail scan data for Hass avocados, including:
- National retail volume (units)
- Prices

## Key Steps in the Notebook

### 1. Setup and Configuration
- Installation of Python 3.8 and necessary libraries such as `scipy`, `statsmodels`, and `numpy`.
- Environment configuration using `pip` to ensure dependencies are installed.

### 2. Exploratory Data Analysis (EDA)
- Data loading and initial cleaning steps.
- Exploration of key features, such as sales volume and price trends, across time and regions.

### 3. Forecasting Techniques
- **ARIMA Model**:
  - Statistical modeling technique used to predict avocado prices.
  - Focuses on identifying trends and seasonality in the data.
- **Prophet Model**:
  - A forecasting tool designed for time-series data.
  - Used to compare and validate predictions against the ARIMA model.

### 4. Model Evaluation
- Evaluation of both ARIMA and Prophet models using accuracy metrics.
- Comparison of forecasted trends to assess model performance.

### 5. Insights and Recommendations
#### Insights:
1. **Trends:**
   - Avocado prices show clear seasonal fluctuations, with higher prices observed during certain months.
   - Sales volumes increase during price dips, indicating price sensitivity among consumers.
   
2. **Regional Variations:**
   - Certain regions, such as the West and Northeast, have consistently higher sales volumes compared to other regions.
   - Price variations across regions suggest differences in consumer behavior and purchasing power.

3. **Impact of Promotions:**
   - Promotions significantly boost sales volume, especially during high-price periods.
   - Channels with promotional offers show a higher conversion rate compared to regular pricing.

4. **Forecast Accuracy:**
   - The Prophet model performed better at capturing seasonality and long-term trends compared to ARIMA, making it more suitable for business planning.
   - ARIMA was effective for short-term forecasting but struggled with complex seasonal patterns.

#### Recommendations:
1. **Dynamic Pricing Strategy:**
   - Implement dynamic pricing to capitalize on seasonal demand fluctuations.
   - Lower prices slightly during peak demand seasons to maximize volume while maintaining profitability.

2. **Regional Focus:**
   - Allocate more resources to high-performing regions like the West and Northeast to maximize returns.
   - Explore targeted campaigns to boost sales in underperforming regions.

3. **Promotion Timing:**
   - Schedule promotions during periods of high prices to counteract potential dips in sales volume.
   - Use historical data to predict optimal timing for discounts and promotional offers.

4. **Inventory and Supply Chain Optimization:**
   - Align inventory planning with predicted price and sales trends to minimize overstocking or shortages.
   - Use Prophet’s longer-term forecasts to ensure supply chain readiness for peak demand periods.

5. **Marketing Budget Allocation:**
   - Prioritize marketing budgets for regions and time periods identified as high-impact by the forecasting models.
   - Focus on digital campaigns in high-performing regions to further boost engagement and sales.

6. **Adopt Prophet for Long-Term Planning:**
   - Utilize Prophet's forecasting capabilities for strategic decisions, such as new market entry or long-term budgeting.
   - Use ARIMA for short-term operational adjustments and daily/weekly sales predictions.


## Tools and Libraries
- **Programming Language**: Python
- **Libraries**:
  - `scipy`, `statsmodels`, `numpy`
  - ARIMA and Prophet forecasting libraries
