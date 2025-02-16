# Import library
import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Judul Utama
st.title('Used Car Price Predictor')
st.text('This web can be used to predict the price of a used car')

# Menambahkan sidebar
st.sidebar.header("Please input car features")

def create_user_input():
    
    # Numerical Features
    year = st.sidebar.slider('Year', min_value=1963, max_value=2021, value=2015)
    engine_size = st.sidebar.slider('Engine Size', min_value=1.0, max_value=9.0, value=2.0, step=0.1)
    mileage = st.sidebar.number_input('Mileage', min_value=100, max_value=1_000_000, value=50_000)

    # Categorical Features
    car_type = st.sidebar.selectbox('Car Type', [
        'Yukon', 'Optima', 'CX3', 'Cayenne S', 'Sonata', 'Avalon', 'C300', 'Land Cruiser', 'LS', 'FJ',
        'Tucson', 'Sunny', 'Pajero', 'Azera', 'Focus', '5', 'Spark', 'Pathfinder', 'Accent', 'ML', 'Corolla',
        'Tahoe', 'A', 'Altima', 'Expedition', 'Senta fe', 'Liberty', 'X', 'Land Cruiser Pickup', 'VTC',
        'Malibu', 'The 5', 'Patrol', 'Grand Cherokee', 'Camry', 'Previa', 'SEL', 'MKZ', 'Datsun', 'Hilux',
        'Edge', '6', 'Innova', 'Navara', 'G80', 'Carnival', 'Suburban', 'Camaro', 'Accord', 'Taurus', 'Elantra',
        'Flex', 'S', 'Cerato', 'Furniture', 'Murano', 'Land Cruiser 70', '3', 'Pick up', 'Charger', 'H6', 'Hiace',
        'Fusion', 'Aveo', 'CX9', 'Yaris', 'Sierra', 'Durango', 'CT-S', 'Sylvian Bus', 'ES', 'Navigator', 'Opirus',
        'The 7', 'Creta', 'CS35', 'The 3', 'Sedona', 'Victoria', 'Prestige', 'Safrane', 'Cores', 'Cadenza',
        "D'max", 'Silverado', 'Rio', 'Maxima', 'X-Trail', 'Cruze', 'Seven', 'Prado', 'Caprice', 'Grand Marquis',
        'LX', 'C', 'Impala', 'QX', 'Blazer', 'H1', 'Rav4', 'The M', 'Genesis', 'Traverse', 'Civic', 'Echo Sport',
        'Challenger', 'CL', 'Wrangler', 'A6', 'Dokker', 'CX5', 'Mohave', 'Explorer', 'Rush', 'Sentra', 'Range Rover',
        'Cherokee', 'Copper', 'Veloster', 'E', 'IS', 'Fluence', 'Vego', 'Ciocca', 'Marquis', 'Q', 'F3', 'Kona', 'UX',
        'Beetle', 'Lancer', 'Van R', 'Mustang', 'CS35 Plus', 'DB9', 'Sorento', 'APV', 'Viano', 'EC7', 'Safari',
        'Cadillac', 'Duster', 'RX', 'Platinum', 'Carenz', 'Avanza', 'Emgrand', 'D-MAX', 'Dyna', 'Z', 'Coupe S',
        'Odyssey', 'Panamera', 'Juke', 'Sportage', 'F150', 'C200', 'Attrage', 'GS', 'X-Terra', 'Picanto', 'CT5',
        'KICKS', 'Gran Max', 'Cayman', 'A8', 'Optra', 'GLC', 'Other', 'Montero', '300', 'A3', 'Touareg', 'Passat',
        'Delta', 'Acadia', 'H3', 'GS3', 'Coupe', 'Cayenne Turbo', 'Colorado', 'Vitara', 'Kaptiva', 'Nativa', 'CLS',
        'LF X60', 'Aurion', 'Koleos', 'Abeka', 'Flying Spur', 'Pilot', 'L200', 'Ranger', 'Escalade', 'A7',
        'Quattroporte', 'Compass', 'Bus Urvan', 'Macan', 'Azkarra', 'GL', 'City', 'Symbol', 'Ertiga', 'RX5',
        'Envoy', 'CT6', 'Fleetwood', 'Tiggo', 'Q5', 'A4', 'XJ', 'H2', 'HS', 'Seltos', 'RX8', '301', 'EC8', '3008',
        'Suvana', 'Prius', 'Cayenne', 'Eado', 'Royal', 'NX', 'CS75', 'F-Pace', 'Coolray', 'CS85', 'Jimny', 'GC7',
        '360', 'A5', 'S300', 'Superb', 'Ram', 'Terrain', 'Cressida', '500', 'Armada', 'Logan', '5008', 'Tiguan',
        'Golf', 'CS95', 'S5', 'M', 'Daily', 'Nitro', 'Mini Van', 'Pegas', 'Grand Vitara', 'FX', 'L300', 'Coaster',
        'Montero2', 'Z370', 'Bus County', 'Stinger', 'SRT', 'CLA', 'K5', 'CT4', 'CC', 'ASX', 'Carens', 'XT5',
        'Tuscani', '4Runner', 'ATS', 'CRV', 'The 4', 'HRV', 'X7', 'GX', 'X40', 'Q7', 'ZS', 'G70', 'Megane', 'Power',
        'B50', 'Town Car', 'GLE', 'Van', '2', 'i40', 'XF', 'Doblo', 'MKX', 'Jetta', 'Soul', 'Lumina', 'Dzire',
        'Avante', 'Z350', 'CX7', 'Countryman', 'Prestige Plus', 'MKS', 'Milan', 'Savana', 'The 6'
    ])
    
    make = st.sidebar.selectbox('Make', [
        'GMC', 'Kia', 'Mazda', 'Porsche', 'Hyundai', 'Toyota', 'Chrysler', 'Lexus', 'Nissan', 'Mitsubishi', 
        'Ford', 'MG', 'Chevrolet', 'Mercedes', 'Jeep', 'BMW', 'Lincoln', 'Genesis', 'Honda', 'Zhengzhou', 
        'Dodge', 'HAVAL', 'Cadillac', 'Changan', 'Renault', 'Suzuki', 'Mercury', 'INFINITI', 'Audi', 
        'Land Rover', 'MINI', 'BYD', 'Volkswagen', 'Victory Auto', 'Aston Martin', 'Geely', 'Classic', 
        'Isuzu', 'Daihatsu', 'Other', 'Hummer', 'GAC', 'Lifan', 'Bentley', 'Maserati', 'Chery', 'Jaguar', 
        'Peugeot', 'Foton', 'Škoda', 'Fiat', 'Iveco', 'FAW', 'Great Wall'
    ])
    
    region = st.sidebar.selectbox('Region', [
        'Riyadh', 'Jeddah', 'Dammam', 'Makkah', 'Medina', 'Tabouk', 'Yanbu', 'Hail', 'Abha'
    ])
    
    gear_type = st.sidebar.radio('Gear Type', ['Automatic', 'Manual'])
    
    origin = st.sidebar.radio('Origin', ['Saudi', 'Gulf Arabic', 'Other', 'Unknown'])
    
    options = st.sidebar.radio('Options', ['Full', 'Semi Full', 'Standard'])

    # Creating a dictionary with user input
    user_data = {
        'Year': year,
        'Engine_Size': engine_size,
        'Mileage': mileage,
        'Type': car_type,
        'Make': make,
        'Region': region,
        'Gear_Type': gear_type,
        'Origin': origin,
        'Options': options
    }
    
    # Convert the dictionary into a pandas DataFrame (for a single row)
    user_data_df = pd.DataFrame([user_data])
    
    return user_data_df

# Get car data from user
data_car = create_user_input()

# Membuat 2 kontainer
col1, col2 = st.columns(2)

# Kiri
with col1:
    st.subheader("Car's Features")
    st.write(data_car.transpose())

# Load model
with open('final_model.sav', 'rb') as f:
    model_loaded = pickle.load(f)
    
# Predict price
predicted_price = model_loaded.predict(data_car)
    
# Menampilkan hasil prediksi

# Bagian kanan (col2)
with col2:
    st.subheader('Predicted Price')
    st.write(f"The estimated price of this car is: **{predicted_price[0]:,.2f} SAR**")
