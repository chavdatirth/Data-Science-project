import json

notebook_path = 'House_Prices_Prediction.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Robustly remove previous interactive cells
to_keep = []
for cell in nb['cells']:
    source_text = str(cell.get('source', ''))
    if 'Interactive' in source_text or 'Enter Custom House Features' in source_text or '11. Interactive Live Prediction' in source_text:
        continue
    to_keep.append(cell)
nb['cells'] = to_keep

new_markdown_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 11. Custom Interactive Price Prediction (Demo!)\n",
        "We're going to dynamically predict the price of a house using the following key features requested:\n",
        "- **House Style**\n",
        "- **Overall Quality**\n",
        "- **Overall Condition**\n",
        "- **Year Built**\n",
        "- **Total Rooms**\n",
        "\n",
        "Just run this cell, input your values, and the model will intelligently fill in all 70+ other fields automatically!"
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
        "print(\"=========================================\")\n",
        "print(\"    🏠 ENTER CUSTOM HOUSE FEATURES 🏠    \")\n",
        "print(\"=========================================\")\n",
        "try:\n",
        "    # 1. HouseStyle\n",
        "    h_style = input(\"House Style (e.g. 1Story, 2Story, 1.5Fin)? [Press Enter for '1Story']: \").strip()\n",
        "    house_style = h_style if h_style else '1Story'\n",
        "\n",
        "    # 2. OverallQual\n",
        "    q_qual = input(\"Overall Quality (1-10)? [Press Enter for 7]: \").strip()\n",
        "    overall_qual = int(q_qual) if q_qual else 7\n",
        "\n",
        "    # 3. OverallCond\n",
        "    q_cond = input(\"Overall Condition (1-10)? [Press Enter for 5]: \").strip()\n",
        "    overall_cond = int(q_cond) if q_cond else 5\n",
        "    \n",
        "    # 4. YearBuilt\n",
        "    y = input(\"Year Built (e.g. 2005)? [Press Enter for 2005]: \").strip()\n",
        "    year_built = int(y) if y else 2005\n",
        "    \n",
        "    # 5. TotRmsAbvGrd\n",
        "    r = input(\"Total Rooms Above Ground? [Press Enter for 6]: \").strip()\n",
        "    tot_rms = int(r) if r else 6\n",
        "    \n",
        "    # INJECT VALUES\n",
        "    custom_house.at[0, 'HouseStyle'] = house_style\n",
        "    custom_house.at[0, 'OverallQual'] = overall_qual\n",
        "    custom_house.at[0, 'OverallCond'] = overall_cond\n",
        "    custom_house.at[0, 'YearBuilt'] = year_built\n",
        "    custom_house.at[0, 'TotRmsAbvGrd'] = tot_rms\n",
        "    \n",
        "    pred_log = loaded_model.predict(custom_house)\n",
        "    pred_usd = np.expm1(pred_log)[0]\n",
        "    \n",
        "    print(\"\\n\" + \"*\"*45)\n",
        "    print(f\" 🌟 PREDICTED HOUSE PRICE: ${pred_usd:,.2f} 🌟 \")\n",
        "    print(\"*\"*45 + \"\\n\")\n",
        "except Exception as e:\n",
        "    print(f\"\\n⚠️ An error occurred computing the prediction:\\n{e}\")\n"
    ]
}

nb['cells'].append(new_markdown_cell)
nb['cells'].append(new_code_cell)

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Updated notebook cells successfully.")
