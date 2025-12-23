FEATURE_COLUMNS = {
    "numeric": [
        "zone_avg_duration",
        "zone_trip_count"
    ],
    "categorical": [
        "PUlocationID",
        "DOlocationID",
        "pickup_hour",
        "pickup_weekday",
        "pickup_day",
        "pickup_week_of_year",
        "is_weekend",
        "is_peak_hour",
        "hour_zone"
    ]
}

ENTITY_KEY = "PUlocationID"
