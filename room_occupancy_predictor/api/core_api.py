from flask import Flask, request, abort, jsonify, make_response
from os import environ
from functools import wraps
from room_occupancy_predictor.room_occupancy_predictor import RoomOccupancyPredictor
import datetime


class MyFlaskApp:
    TRAIN_DATA_PATH = 'data/device_activations.csv'

    def __init__(self):
        self.room_occupancy_predictor = RoomOccupancyPredictor(self.TRAIN_DATA_PATH)
        self.app = self.__init_app()

        @self.app.route("/predict", methods=["GET"])
        @self.check_token
        def predict():
            timestamp = datetime.datetime.now()
            response = self.room_occupancy_predictor.create_output_df(timestamp).to_dict('records')
            return jsonify(response)

    def check_token(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            token = request.args.get("private_token")
            if token != self.app.config["PRIVATE_TOKEN"]:
                return abort(make_response(jsonify(message="Wrong token"), 403))
            return func(*args, **kwargs)
        return wrapper

    @staticmethod
    def __init_app():
        app = Flask(__name__)
        app.config["TOKEN"] = environ["TOKEN"]
        return app