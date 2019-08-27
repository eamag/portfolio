FROM continuumio/miniconda3
RUN pip install loguru scikit-learn catboost numpy pandas flask
EXPOSE 5000
COPY . /room_occupancy_predictor
WORKDIR /room_occupancy_predictor
CMD ["python", "run_app.py"]