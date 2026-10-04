import pandas as pd
import numpy as np
import pickle
import warnings
import sys
warnings.filterwarnings('ignore')

print("\n" + "="*50)
print("     🏠 INDIAN REAL ESTATE PREDICTOR 🏠    ")
print("="*50 + "\n")

try:
    with open('indian_house_price_model.pkl', 'rb') as f:
        loaded_model = pickle.load(f)
except FileNotFoundError:
    print("Error: indian_house_price_model.pkl not found! Please run the notebook first.")
    sys.exit(1)

reference_columns = ['City', 'Area_SqFt', 'BHK', 'Property_Age_Years', 'Furnishing', 'Parking', 'RERA_Approved']
custom_house = pd.DataFrame(columns=reference_columns)
custom_house.loc[0] = np.nan

try:
    print("City Options: [1] Mumbai, [2] Delhi, [3] Bangalore, [4] Hyderabad, [5] Chennai, [6] Pune, [7] Ahmedabad")
    c = input("Enter City Option (1-7)? [Press Enter for 3:Bangalore]: ").strip()
    city_map = {'1':'Mumbai', '2':'Delhi', '3':'Bangalore', '4':'Hyderabad', '5':'Chennai', '6':'Pune', '7':'Ahmedabad'}
    city = city_map.get(c, 'Bangalore')

    a = input("\nArea (in SqFt)? [Press Enter for 1200]: ").strip()
    area = int(a) if a else 1200

    b = input("\nBHK (1-6)? [Press Enter for 3]: ").strip()
    bhk = int(b) if b else 3
    
    print("\nFurnishing Options: [1] Unfurnished, [2] Semi-Furnished, [3] Fully-Furnished")
    f = input("Enter Furnishing (1-3)? [Press Enter for 3:Fully-Furnished]: ").strip()
    furn_map = {'1':'Unfurnished', '2':'Semi-Furnished', '3':'Fully-Furnished'}
    furnishing = furn_map.get(f, 'Fully-Furnished')
    
    age_input = input("\nProperty Age in Years? [Press Enter for 2]: ").strip()
    property_age = int(age_input) if age_input else 2
    
    print("\nParking Options: [1] None, [2] Open, [3] Covered")
    p = input("Enter Parking (1-3)? [Press Enter for 3:Covered]: ").strip()
    park_map = {'1':'None', '2':'Open', '3':'Covered'}
    parking = park_map.get(p, 'Covered')
    
    print("\nRERA Approved Options: [1] Yes, [2] No")
    r = input("Enter RERA (1-2)? [Press Enter for 1:Yes]: ").strip()
    rera_map = {'1':'Yes', '2':'No'}
    rera_approved = rera_map.get(r, 'Yes')
    
    custom_house.at[0, 'City'] = city
    custom_house.at[0, 'Area_SqFt'] = area
    custom_house.at[0, 'BHK'] = bhk
    custom_house.at[0, 'Property_Age_Years'] = property_age
    custom_house.at[0, 'Furnishing'] = furnishing
    custom_house.at[0, 'Parking'] = parking
    custom_house.at[0, 'RERA_Approved'] = rera_approved
    
    pred_lakhs = loaded_model.predict(custom_house)[0]
    
    print("\n" + "*"*50)
    print(f" 🇮🇳 PREDICTED PRICE: ₹{pred_lakhs:,.2f} Lakhs ")
    if pred_lakhs > 100:
        crores = pred_lakhs / 100.0
        print(f" 🇮🇳 Or Approximately: ₹{crores:,.2f} Crores ")
    print("*"*50 + "\n")
    
    input("Press Enter to exit...")
except Exception as e:
    print(f"\n⚠️ An error occurred computing the prediction:\n{e}")
