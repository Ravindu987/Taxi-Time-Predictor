import requests


def test_service():
    prediction = requests.post(
        "http://127.0.0.1:3000/predict",
        headers={"content-type": "application/json"},
        data='{ "vendor_id": 1, "pickup_datetime": "2016-06-30 23:59:58", "passenger_count": 1, "pickup_longitude": -73.9881286621094, "pickup_latitude": 40.7320289611816, "dropoff_longitude": -73.9901733398438, "dropoff_latitude": 40.7566795349121, "store_and_fwd_flag": "N"}',
    ).text

    prediction = float(prediction[1:-1])
    assert prediction >= 0 and prediction <= 10000
