import json

notebook_path = 'House_Prices_Prediction.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

new_markdown_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 11. Interactive Live Prediction (Demo!)\n",
        "Run this cell to input custom house parameters and get a live price prediction! For any parameters we don't manually input, the model will intelligently fill in the averages automatically."
    ]
}

new_code_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "import pandas as pd\n",
        "import numpy as np\n",
        "import pickle\n",
        "import warnings\n",
        "warnings.filterwarnings('ignore')\n",
        "\n",
        "with open('best_house_price_model.pkl', 'rb') as f:\n",
        "    loaded_model = pickle.load(f)\n",
        "\n",
        "reference_columns = X.columns\n",
        "custom_house = pd.DataFrame(columns=reference_columns)\n",
        "custom_house.loc[0] = np.nan\n",
        "\n",
        "print(\"=== Enter Custom House Features ===\")\n",
        "try:\n",
        "    q = input(\"Overall Quality (1-10)? [Press Enter for default 7]: \")\n",
        "    overall_qual = int(q) if q.strip() else 7\n",
        "    \n",
        "    a = input(\"Above Ground Living Area sqft? [Press Enter for default 1500]: \")\n",
        "    gr_liv_area = float(a) if a.strip() else 1500.0\n",
        "    \n",
        "    y = input(\"Year Built? [Press Enter for default 2005]: \")\n",
        "    year_built = int(y) if y.strip() else 2005\n",
        "    \n",
        "    g = input(\"Garage Capacity (in cars)? [Press Enter for default 2]: \")\n",
        "    garage_cars = float(g) if g.strip() else 2.0\n",
        "    \n",
        "    custom_house.at[0, 'OverallQual'] = overall_qual\n",
        "    custom_house.at[0, 'GrLivArea'] = gr_liv_area\n",
        "    custom_house.at[0, 'YearBuilt'] = year_built\n",
        "    custom_house.at[0, 'GarageCars'] = garage_cars\n",
        "    \n",
        "    pred_log = loaded_model.predict(custom_house)\n",
        "    pred_usd = np.expm1(pred_log)[0]\n",
        "    \n",
        "    print(\"\\n========================================\")\n",
        "    print(f\" 🏡 PREDICTED HOUSE PRICE: ${pred_usd:,.2f} \")\n",
        "    print(\"========================================\\n\")\n",
        "except Exception as e:\n",
        "    print(f\"An error occurred:\\n{e}\")\n"
    ]
}

# Avoid duplicating if ran twice
if "Interactive Live Prediction" not in str(nb['cells'][-1]):
    nb['cells'].append(new_markdown_cell)
    nb['cells'].append(new_code_cell)

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Appended cells successfully.")
