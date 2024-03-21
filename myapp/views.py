from django.shortcuts import render, HttpResponse
from django.http import JsonResponse
import requests
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import datetime
import json
from collections import defaultdict


# Create your views here.
def index(request):
    context = {
        'variable': ''
    }
    if request.method == 'POST':
        # Retrieve the values of product name and date from the POST request
        product_name = request.POST.get('productName')
        date = request.POST.get('date')
        
        # Log the product name and date
        print("Product Name:", product_name)
        print("Date:", date)

 

        url = "https://product-price-history.p.rapidapi.com/price-history"

        querystring = {"country_iso2":"nl","gtin":"0195908483175","last_x_months":"24"}

        headers = {
            "X-RapidAPI-Key": "d4d743c909msh724f9b0cd56c97cp180c3fjsne78028dc52f6",
            "X-RapidAPI-Host": "product-price-history.p.rapidapi.com"
        }

        response = requests.get(url, headers=headers, params=querystring)

        priceHistory = response.json()

        print(response.json())


        #TESTING CODE FOR COST PREDICTION

        # Assuming you have the JSON data stored in a variable named 'json_data'
        # Load data from JSON
        data = priceHistory

        # Extract timestamps and average prices
        timestamps = []
        avg_prices = []

        for timestamp, info in data.items():
            timestamps.append(info['key'])  # Assuming 'key' is the numerical representation of the timestamp
            avg_prices.append(info['avg_price_in_cents'])

        # Convert timestamps into a 2D array
        X = np.array(timestamps).reshape(-1, 1)

        # Convert average prices into a 1D array
        y = np.array(avg_prices)

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train linear regression model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Evaluate model
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print("Mean Squared Error:", mse)

        # Function to predict price for a specific date
        def predict_price_for_date(date):
            # Convert date to timestamp
            timestamp = int(date.timestamp())
            # Predict price using the model
            predicted_price = model.predict(np.array([[timestamp]]))
            return predicted_price

        # Example: Predict price for August 19, 2025
        input_date = datetime.datetime(2025, 8, 19)
        predicted_price = predict_price_for_date(input_date)

        predicted_price = predicted_price * 0.0542

        print("Predicted price for", input_date.strftime('%Y-%m-%d'), ":", predicted_price)
        
        # Return a JSON response indicating successful logging
        # return JsonResponse({'message': 'Data logged successfully'})

        # MINIMUM AND MAXIMUM PRICE MONTH PREDICTION

        # Group data by month
        print("Data is fine here : ", data)
        monthly_data = defaultdict(list)
        for date_str, info in data.items():
            month = date_str[:7]
            monthly_data[month].append(info)
        
        # Calculate total purchases and average price for each month
        result = {}
        for month, purchases in monthly_data.items():
            total_purchases = sum(info['data_points'] for info in purchases)
            avg_price = sum(info['avg_price_in_cents'] * info['data_points'] for info in purchases) / total_purchases
            result[month] = {'total_purchases': total_purchases, 'avg_price': avg_price}
        
        # Find the month with maximum purchases and lowest average price
        max_purchases_month = max(result, key=lambda m: result[m]['total_purchases'])
        lowest_avg_price_month = min(result, key=lambda m: result[m]['avg_price'])

        print("Month with maximum purchases:", max_purchases_month)
        print("Month with lowest average price:", lowest_avg_price_month)

        context = {
            'variable': predicted_price,
            'max_purchases_month': max_purchases_month,
            'lowest_avg_price_month': lowest_avg_price_month
        }
    else:
        context = {
            'variable': ''
        }
        
    return render(request, 'index.html', context)

def about(request):
    return HttpResponse("This is the about page")

def details(request):
    return HttpResponse("This is the details page")
