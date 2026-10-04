import pandas as pd
import numpy as np
import os

np.random.seed(42)

def generate_houses(n_samples):
    # Cities and base prices/multipliers
    cities = ['Mumbai', 'Delhi', 'Bangalore', 'Hyderabad', 'Chennai', 'Pune', 'Ahmedabad']
    city_base = {'Mumbai': 50, 'Delhi': 30, 'Bangalore': 25, 'Hyderabad': 20, 'Chennai': 22, 'Pune': 20, 'Ahmedabad': 25}
    city_sqft_rate = {'Mumbai': 15000, 'Delhi': 8000, 'Bangalore': 7000, 'Hyderabad': 5500, 'Chennai': 6000, 'Pune': 6500, 'Ahmedabad': 6000}
    
    city = np.random.choice(cities, n_samples)
    
    # Area between 400 and 4500
    area = np.random.randint(400, 4500, n_samples)
    
    # BHK normally scales with Area
    bhk = np.clip((area // 600) + np.random.randint(0, 2, n_samples), 1, 6)
    
    # Property Age 0 to 30 years
    age = np.random.randint(0, 31, n_samples)
    
    # Furnishing
    furnishing = np.random.choice(['Unfurnished', 'Semi-Furnished', 'Fully-Furnished'], n_samples, p=[0.4, 0.4, 0.2])
    furnish_val = {'Unfurnished': 0, 'Semi-Furnished': 5, 'Fully-Furnished': 12}
    
    # Parking
    parking = np.random.choice(['None', 'Open', 'Covered'], n_samples, p=[0.2, 0.4, 0.4])
    parking_val = {'None': 0, 'Open': 2, 'Covered': 6}
    
    # RERA Approved
    rera = np.random.choice(['Yes', 'No'], n_samples, p=[0.8, 0.2])
    
    # Calculate Prices
    prices = []
    for i in range(n_samples):
        # 1. Base price for the city
        price = city_base[city[i]]
        
        # 2. Add sqft valuation (conversion to Lakhs)
        sqft_val = (area[i] * city_sqft_rate[city[i]]) / 100000 
        price += sqft_val
        
        # 3. BHK premium
        price += (bhk[i] * 4.0)
        
        # 4. Depreciation based on age
        price -= (age[i] * 0.8)
        
        # 5. Amenities
        price += furnish_val[furnishing[i]]
        price += parking_val[parking[i]]
        
        # 6. RERA Premium
        if rera[i] == 'Yes':
            price *= 1.05
            
        # 7. Add noise to make it realistic for ML
        noise = np.random.normal(0, price * 0.1) # 10% standard deviation noise
        price += noise
        
        # Ensure floor price
        if price < 10:
            price = 10 + np.random.rand() * 5
            
        prices.append(round(price, 2))
        
    df = pd.DataFrame({
        'Property_ID': range(1, n_samples + 1),
        'City': city,
        'Area_SqFt': area,
        'BHK': bhk,
        'Property_Age_Years': age,
        'Furnishing': furnishing,
        'Parking': parking,
        'RERA_Approved': rera,
        'Price_Lakhs': prices
    })
    
    return df

print("Generating Indian housing dataset...")

# Generate 2500 training points and 500 test points
train_df = generate_houses(2500)
test_df = generate_houses(500)

# Shift test IDs
test_df['Property_ID'] = range(2501, 3001)

# Test DF doesn't have the target
test_actual_prices = test_df[['Property_ID', 'Price_Lakhs']] # Just saving truth for any validation later
test_df = test_df.drop(columns=['Price_Lakhs'])

os.makedirs('data', exist_ok=True)
train_df.to_csv('data/indian_train.csv', index=False)
test_df.to_csv('data/indian_test.csv', index=False)

print("Datasets 'indian_train.csv' and 'indian_test.csv' successfully created in data/ folder!")
