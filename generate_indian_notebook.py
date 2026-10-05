import json

cells = []

cells.append({
    "cell_type": "markdown", "metadata": {},
    "source": [
        "# Indian Housing Price Prediction\n",
        "### Specialized Model handling BHK, Area, City, and Furnishing with Interactive GUI"
    ]
})

cells.append({
    "cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
    "source": [
        "import pandas as pd\n",
        "import numpy as np\n",
        "import pickle\n",
        "import warnings\n",
        "warnings.filterwarnings('ignore')\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.pipeline import Pipeline\n",
        "from sklearn.compose import ColumnTransformer\n",
        "from sklearn.impute import SimpleImputer\n",
        "from sklearn.preprocessing import StandardScaler, OneHotEncoder\n",
        "from sklearn.linear_model import Ridge\n",
        "from sklearn.metrics import mean_squared_error, r2_score\n"
    ]
})

cells.append({
    "cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
    "source": [
        "train = pd.read_csv('data/indian_train.csv')\n",
        "test = pd.read_csv('data/indian_test.csv')\n",
        "print('Train Shape:', train.shape)\n",
        "print('Test Shape:', test.shape)\n"
    ]
})

cells.append({
    "cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
    "source": [
        "X = train.drop(columns=['Property_ID', 'Price_Lakhs'])\n",
        "y = train['Price_Lakhs']\n",
        "X_test = test.drop(columns=['Property_ID'])\n",
        "\n",
        "numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()\n",
        "categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()\n",
        "\n",
        "num_pipeline = Pipeline(steps=[\n",
        "    ('imputer', SimpleImputer(strategy='median')),\n",
        "    ('scaler', StandardScaler())\n",
        "])\n",
        "cat_pipeline = Pipeline(steps=[\n",
        "    ('imputer', SimpleImputer(strategy='most_frequent')),\n",
        "    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))\n",
        "])\n",
        "preprocessor = ColumnTransformer(transformers=[\n",
        "    ('num', num_pipeline, numerical_cols),\n",
        "    ('cat', cat_pipeline, categorical_cols)\n",
        "])\n"
    ]
})

cells.append({
    "cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
    "source": [
        "X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)\n",
        "model_pipeline = Pipeline(steps=[\n",
        "    ('preprocessor', preprocessor),\n",
        "    ('model', Ridge(alpha=1.0))\n",
        "])\n",
        "model_pipeline.fit(X_train, y_train)\n",
        "preds = model_pipeline.predict(X_val)\n",
        "rmse = np.sqrt(mean_squared_error(y_val, preds))\n",
        "r2 = r2_score(y_val, preds)\n",
        "print(f\"Validation RMSE: {rmse:.2f} Lakhs\")\n",
        "print(f\"R2 Score: {r2:.4f}\")\n"
    ]
})

cells.append({
    "cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
    "source": [
        "with open('indian_house_price_model.pkl', 'wb') as f:\n",
        "    pickle.dump(model_pipeline, f)\n",
        "print('Model saved to indian_house_price_model.pkl')\n"
    ]
})

cells.append({
    "cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
    "source": [
        "test_preds = model_pipeline.predict(X_test)\n",
        "submission = pd.DataFrame({'Property_ID': test['Property_ID'], 'Price_Lakhs': test_preds})\n",
        "submission.to_csv('indian_submission.csv', index=False)\n",
        "print('Predictions saved to indian_submission.csv')\n"
    ]
})

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 🖥️ Interactive GUI: Indian Real Estate Price Estimator\n",
        "Use the visual control panel below with dropdown menus and sliders to dynamically estimate property valuations in real-time."
    ]
})

gui_code = """import ipywidgets as widgets
from IPython.display import display, HTML
import pandas as pd
import numpy as np
import pickle

# Load the saved model pipeline
with open('indian_house_price_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# Define Interactive UI Components
city_dd = widgets.Dropdown(
    options=['Mumbai', 'Delhi', 'Bangalore', 'Hyderabad', 'Chennai', 'Pune', 'Ahmedabad'],
    value='Bangalore',
    description='🏙️ City:',
    style={'description_width': '130px'},
    layout=widgets.Layout(width='380px')
)

area_slider = widgets.IntSlider(
    value=1200,
    min=300,
    max=10000,
    step=50,
    description='📐 Area (SqFt):',
    style={'description_width': '130px'},
    layout=widgets.Layout(width='380px')
)

bhk_dd = widgets.Dropdown(
    options=[1, 2, 3, 4, 5, 6],
    value=3,
    description='🛏️ BHK:',
    style={'description_width': '130px'},
    layout=widgets.Layout(width='380px')
)

furn_dd = widgets.Dropdown(
    options=['Unfurnished', 'Semi-Furnished', 'Fully-Furnished'],
    value='Fully-Furnished',
    description='🛋️ Furnishing:',
    style={'description_width': '130px'},
    layout=widgets.Layout(width='380px')
)

age_slider = widgets.IntSlider(
    value=2,
    min=0,
    max=50,
    step=1,
    description='⏳ Age (Years):',
    style={'description_width': '130px'},
    layout=widgets.Layout(width='380px')
)

park_dd = widgets.Dropdown(
    options=['None', 'Open', 'Covered'],
    value='Covered',
    description='🚗 Parking:',
    style={'description_width': '130px'},
    layout=widgets.Layout(width='380px')
)

rera_dd = widgets.Dropdown(
    options=['Yes', 'No'],
    value='Yes',
    description='📜 RERA Approved:',
    style={'description_width': '130px'},
    layout=widgets.Layout(width='380px')
)

predict_btn = widgets.Button(
    description=' Calculate Estimated Price',
    button_style='success',
    icon='calculator',
    layout=widgets.Layout(width='380px', height='45px', margin='15px 0 0 0')
)

output_area = widgets.Output()

def on_predict_clicked(b):
    with output_area:
        output_area.clear_output()
        custom_house = pd.DataFrame([{
            'City': city_dd.value,
            'Area_SqFt': area_slider.value,
            'BHK': bhk_dd.value,
            'Property_Age_Years': age_slider.value,
            'Furnishing': furn_dd.value,
            'Parking': park_dd.value,
            'RERA_Approved': rera_dd.value
        }])
        
        pred_lakhs = loaded_model.predict(custom_house)[0]
        
        crores_html = ""
        if pred_lakhs >= 100:
            crores_html = f"<div style='font-size: 16px; color: #4a5568; margin-top: 4px;'>Approx: <b>₹{pred_lakhs/100:.2f} Crores</b></div>"
            
        card_html = f'''
        <div style="border: 2px solid #28a745; border-radius: 12px; padding: 18px; margin-top: 15px; background: linear-gradient(135deg, #f0fff4, #e6fffa); width: 400px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <div style="font-size: 13px; font-weight: bold; color: #276749; text-transform: uppercase; letter-spacing: 1px;">🏠 Estimated Market Valuation</div>
            <div style="font-size: 30px; font-weight: 800; color: #22543d; margin: 8px 0;">₹{pred_lakhs:,.2f} Lakhs</div>
            {crores_html}
            <div style="margin-top: 12px; font-size: 12px; color: #4a5568; border-top: 1px solid #cbd5e0; padding-top: 8px;">
                📍 <b>{city_dd.value}</b> &nbsp;|&nbsp; 📐 <b>{area_slider.value} sq.ft</b> &nbsp;|&nbsp; 🛏️ <b>{bhk_dd.value} BHK</b><br>
                🛋️ <b>{furn_dd.value}</b> &nbsp;|&nbsp; 🚗 <b>{park_dd.value} Parking</b> &nbsp;|&nbsp; 📜 <b>RERA: {rera_dd.value}</b>
            </div>
        </div>
        '''
        display(HTML(card_html))

predict_btn.on_click(on_predict_clicked)

# Render GUI Form
title_html = widgets.HTML('''
<div style="background: linear-gradient(135deg, #1e3a8a, #3b82f6); color: white; padding: 16px; border-radius: 10px; width: 400px; margin-bottom: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
    <h3 style="margin: 0; font-family: sans-serif; font-size: 18px;">🏠 Indian Real Estate Valuation GUI</h3>
    <p style="margin: 4px 0 0 0; font-size: 12px; opacity: 0.9;">Adjust parameters below and click calculate</p>
</div>
''')

gui_panel = widgets.VBox([
    title_html,
    city_dd,
    area_slider,
    bhk_dd,
    furn_dd,
    age_slider,
    park_dd,
    rera_dd,
    predict_btn,
    output_area
], layout=widgets.Layout(padding='15px', border='1px solid #cbd5e1', border_radius='12px', width='440px', background_color='#f8fafc'))

display(gui_panel)
"""

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [line + "\n" for line in gui_code.splitlines()]
})

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

with open('Indian_House_Prices_Prediction.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print("Indian notebook successfully updated with GUI!")
