import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib

df = pd.read_csv("house_data.csv")

features = ['Area', 'Bedrooms', 'Bathrooms', 'Parking', 'Floors',
            'Age', 'Balconies', 'Near_Metro', 'School_Distance', 'City']
X = df[features]
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

print("\nEnter house details:")
area = float(input("Area (sqft): "))
bedrooms = int(input("Bedrooms: "))
bathrooms = int(input("Bathrooms: "))
parking = int(input("Parking spaces: "))
floors = int(input("Floors: "))
age = int(input("Age of property (years): "))
balconies = int(input("Balconies: "))
near_metro = int(input("Near Metro? (1=Yes, 0=No): "))
school_distance = float(input("Distance to nearest school (km): "))
city = int(input("City Code (1, 2, 3, etc.): "))

input_data = pd.DataFrame([[area, bedrooms, bathrooms, parking, floors, age,
                            balconies, near_metro, school_distance, city]],
                          columns=features)

predicted_price = model.predict(input_data)
print(f"\nPredicted Price: ₹{predicted_price[0]:,.2f}")

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

joblib.dump(model, "house_price_model.pkl")
print("Model saved as 'house_price_model.pkl'")