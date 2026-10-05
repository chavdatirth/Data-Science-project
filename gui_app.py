import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
import pickle
import os

def load_model():
    model_path = os.path.join(os.path.dirname(__file__), 'indian_house_price_model.pkl')
    if not os.path.exists(model_path):
        messagebox.showerror("Error", f"Model file '{model_path}' not found! Please run the notebook first to generate the model.")
        return None
    with open(model_path, 'rb') as f:
        return pickle.load(f)

class HousePricePredictorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🏠 Indian Real Estate Price Predictor")
        self.root.geometry("520x650")
        self.root.resizable(False, False)
        self.root.configure(bg="#f1f5f9")
        
        self.model = load_model()
        self.setup_ui()
        
    def setup_ui(self):
        # Header Banner
        header = tk.Frame(self.root, bg="#1e3a8a", pady=15)
        header.pack(fill="x")
        
        title_lbl = tk.Label(header, text="🏠 Indian House Price Predictor", font=("Segoe UI", 16, "bold"), fg="white", bg="#1e3a8a")
        title_lbl.pack()
        subtitle_lbl = tk.Label(header, text="Machine Learning Valuation GUI", font=("Segoe UI", 10), fg="#93c5fd", bg="#1e3a8a")
        subtitle_lbl.pack()
        
        # Main Form Container
        form_frame = tk.Frame(self.root, bg="white", padx=25, pady=20, relief="groove", bd=1)
        form_frame.pack(padx=20, pady=15, fill="both", expand=True)
        
        # City
        tk.Label(form_frame, text="City:", font=("Segoe UI", 10, "bold"), bg="white", fg="#334155").grid(row=0, column=0, sticky="w", pady=6)
        self.city_var = tk.StringVar(value="Bangalore")
        cities = ['Mumbai', 'Delhi', 'Bangalore', 'Hyderabad', 'Chennai', 'Pune', 'Ahmedabad']
        self.city_cb = ttk.Combobox(form_frame, textvariable=self.city_var, values=cities, state="readonly", width=25)
        self.city_cb.grid(row=0, column=1, sticky="e", pady=6)
        
        # Area in SqFt
        tk.Label(form_frame, text="Area (Sq.Ft):", font=("Segoe UI", 10, "bold"), bg="white", fg="#334155").grid(row=1, column=0, sticky="w", pady=6)
        self.area_var = tk.IntVar(value=1200)
        self.area_sp = ttk.Spinbox(form_frame, from_=300, to=15000, increment=50, textvariable=self.area_var, width=24)
        self.area_sp.grid(row=1, column=1, sticky="e", pady=6)
        
        # BHK
        tk.Label(form_frame, text="BHK:", font=("Segoe UI", 10, "bold"), bg="white", fg="#334155").grid(row=2, column=0, sticky="w", pady=6)
        self.bhk_var = tk.IntVar(value=3)
        self.bhk_cb = ttk.Combobox(form_frame, textvariable=self.bhk_var, values=[1, 2, 3, 4, 5, 6], state="readonly", width=25)
        self.bhk_cb.grid(row=2, column=1, sticky="e", pady=6)
        
        # Furnishing
        tk.Label(form_frame, text="Furnishing:", font=("Segoe UI", 10, "bold"), bg="white", fg="#334155").grid(row=3, column=0, sticky="w", pady=6)
        self.furn_var = tk.StringVar(value="Fully-Furnished")
        furn_opts = ['Unfurnished', 'Semi-Furnished', 'Fully-Furnished']
        self.furn_cb = ttk.Combobox(form_frame, textvariable=self.furn_var, values=furn_opts, state="readonly", width=25)
        self.furn_cb.grid(row=3, column=1, sticky="e", pady=6)
        
        # Property Age
        tk.Label(form_frame, text="Property Age (Years):", font=("Segoe UI", 10, "bold"), bg="white", fg="#334155").grid(row=4, column=0, sticky="w", pady=6)
        self.age_var = tk.IntVar(value=2)
        self.age_sp = ttk.Spinbox(form_frame, from_=0, to=50, increment=1, textvariable=self.age_var, width=24)
        self.age_sp.grid(row=4, column=1, sticky="e", pady=6)
        
        # Parking
        tk.Label(form_frame, text="Parking:", font=("Segoe UI", 10, "bold"), bg="white", fg="#334155").grid(row=5, column=0, sticky="w", pady=6)
        self.park_var = tk.StringVar(value="Covered")
        park_opts = ['None', 'Open', 'Covered']
        self.park_cb = ttk.Combobox(form_frame, textvariable=self.park_var, values=park_opts, state="readonly", width=25)
        self.park_cb.grid(row=5, column=1, sticky="e", pady=6)
        
        # RERA Approved
        tk.Label(form_frame, text="RERA Approved:", font=("Segoe UI", 10, "bold"), bg="white", fg="#334155").grid(row=6, column=0, sticky="w", pady=6)
        self.rera_var = tk.StringVar(value="Yes")
        rera_opts = ['Yes', 'No']
        self.rera_cb = ttk.Combobox(form_frame, textvariable=self.rera_var, values=rera_opts, state="readonly", width=25)
        self.rera_cb.grid(row=6, column=1, sticky="e", pady=6)
        
        # Predict Button
        predict_btn = tk.Button(form_frame, text="⚡ Predict Estimated Price", font=("Segoe UI", 11, "bold"),
                                bg="#16a34a", fg="white", activebackground="#15803d", activeforeground="white",
                                cursor="hand2", relief="flat", pady=8, command=self.predict_price)
        predict_btn.grid(row=7, column=0, columnspan=2, pady=18, sticky="we")
        
        # Result Card
        self.result_card = tk.Frame(form_frame, bg="#f0fdf4", bd=1, relief="solid", padx=15, pady=12)
        self.result_card.grid(row=8, column=0, columnspan=2, sticky="we")
        
        self.res_title = tk.Label(self.result_card, text="ESTIMATED PROPERTY PRICE", font=("Segoe UI", 9, "bold"), fg="#166534", bg="#f0fdf4")
        self.res_title.pack()
        
        self.res_lakhs = tk.Label(self.result_card, text="₹ 0.00 Lakhs", font=("Segoe UI", 18, "bold"), fg="#15803d", bg="#f0fdf4")
        self.res_lakhs.pack(pady=3)
        
        self.res_crores = tk.Label(self.result_card, text="", font=("Segoe UI", 10), fg="#334155", bg="#f0fdf4")
        self.res_crores.pack()
        
    def predict_price(self):
        if not self.model:
            messagebox.showerror("Error", "Model not loaded!")
            return
        
        try:
            input_df = pd.DataFrame([{
                'City': self.city_var.get(),
                'Area_SqFt': int(self.area_var.get()),
                'BHK': int(self.bhk_var.get()),
                'Property_Age_Years': int(self.age_var.get()),
                'Furnishing': self.furn_var.get(),
                'Parking': self.park_var.get(),
                'RERA_Approved': self.rera_var.get()
            }])
            
            price_lakhs = self.model.predict(input_df)[0]
            
            self.res_lakhs.config(text=f"₹ {price_lakhs:,.2f} Lakhs")
            if price_lakhs >= 100:
                crores = price_lakhs / 100.0
                self.res_crores.config(text=f"Approximately: ₹ {crores:,.2f} Crores")
            else:
                self.res_crores.config(text="")
        except Exception as e:
            messagebox.showerror("Prediction Error", f"Error calculating prediction:\n{e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = HousePricePredictorGUI(root)
    root.mainloop()
