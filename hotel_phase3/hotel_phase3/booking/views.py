import os
import pandas as pd
import joblib

from django.shortcuts import render
from django.conf import settings
from .models import HotelPrediction


def dashboard(request):
    data_path = os.path.join(settings.BASE_DIR, "data", "hotel_booking.csv")
    df = pd.read_csv(data_path)

    total_bookings = len(df)
    canceled_bookings = int(df["is_canceled"].sum())
    cancellation_rate = round((canceled_bookings / total_bookings) * 100, 2)

    hotel_counts = df["hotel"].value_counts().to_dict()

    context = {
        "total_bookings": total_bookings,
        "canceled_bookings": canceled_bookings,
        "cancellation_rate": cancellation_rate,
        "hotel_counts": hotel_counts,
    }

    return render(request, "dashboard.html", context)


def predict(request):
    prediction = None

    if request.method == "POST":
        lead_time = int(request.POST.get("lead_time"))
        adults = int(request.POST.get("adults"))
        stays_in_week_nights = int(request.POST.get("stays_in_week_nights"))

        model_path = os.path.join(
            settings.BASE_DIR,
            "saved_models",
            "hotel_model.joblib"
        )

        model = joblib.load(model_path)

        input_data = [[lead_time, adults, stays_in_week_nights]]
        result = model.predict(input_data)[0]

        # Save prediction using ORM
        HotelPrediction.objects.create(
            lead_time=lead_time,
            adults=adults,
            stays_in_week_nights=stays_in_week_nights,
            prediction=result
        )

        if result == 1:
            prediction = "The booking is likely to be cancelled."
        else:
            prediction = "The booking is not likely to be cancelled."

    return render(request, "predict.html", {"prediction": prediction})