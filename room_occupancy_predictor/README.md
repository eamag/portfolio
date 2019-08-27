The challenge consists of three parts - the first two require code as output, for the third part only a conceptual, text-based answer is needed.

# How to run the solution:
### #1
```bash
./room_occupancy_predictor/room_occupancy_predictor.py <timestamp> <input file csv> <output file csv>
```
P.s. I usually write tests with pytest, but it takes time, and this is test task
### #2
```bash
docker build --rm -f "Dockerfile" -t room_occupancy_predictor:app . && \
docker run -e PRIVATE_TOKEN=token -p 5000:5000 \
-it --rm room_occupancy_predictor:app
```
and then check if its working:
```bash
curl http://localhost:5000/predict?token=token
```

### #3
First we have to decide why do we want to setup infrastructure like this. Lets assume we want to optimize heating consumption.
Incoming data processing can be implemented via collecting the data to the huge PostgreSQL DB, master/slave relationship, multi threaded input etc. 
Batch processing like retraining the models should be scheduled in non-peak hours like midnight, monitoring via ELK stack.
Predictions can be stored in Redis, data processing like agg can be done using Spark. I personally worked with AWS stack for everything in cloud.
To get more detailed solution we have to know how it will be used, i.e. how many requests, how stable are they, how soon the new information should be accounted etc.


# 1) Occupancy prediction model

### Challenge Description

You are provided with time series data from motion sensors (in this case [passive infrared sensors](https://en.wikipedia.org/wiki/Passive_infrared_sensor)). These sensors can be installed in buildings to help determine occupancy of rooms or movement in corridors. The data from these devices can be used to turn lights on and off or to derive optimized heating or ventilation schedules. Judging from the readings, you might know that nobody is usually in the office at a certain time of day so you can turn of the ventilation at this time. 

In our case, sensors were installed in 7 different meeting rooms of an office building and recorded occupancy values in these rooms for two months (July and August 2016). Your objective is to write a *model* which can *predict occupancy for the next 24h after a given timestamp*. The input for the predictor are all sensor readings up to the given timestamp. The output should be a value of 1 if you predict that the room will be occupied and 0 if you predict that it will not be occupied for each of the following 24 hours. You evaluate your predictor based on a score that compares the predicted states with the actual states in the test set.

Are there any points you could think of that could help improve your result (e.g. what if you had more data)?

### Submission details

You should create a program that takes 3 arguments like this:

    ./sample_solution.py <timestamp> <input file csv> <output file csv>

* `timestamp` Is the input time. Your predictions should begin in
the hour following this timestamp.
* `input file` The history of all sensor readings up to the input time. See
format in `data/device_activations.csv`. The sensors in the different rooms are called 'device_[1-7]'.

* `output file` This is where you write your results to. See format in
`data/sample_solution.csv`

### Dummy solution

To help make input / output easy to understand we have included a dummy solution as well as some sample data. If you run it as follows

    ./sample_solution.py '2016-08-31 23:59:59' data/device_activations.csv myresult.csv

Then you should get the file `data/sample_solutions.csv`.


# 2) Dockerized REST API

Now that you have a model that serves predictions, build a simple Python application that predicts the occupancy of one or several rooms for the next 24 hours!

This task consists in *writing a simple REST API* with a single endpoint (GET or POST) that returns those predictions for the next 24 hours as JSON. Ideally, use Docker to containerize your API.

Example usage of your Docker application:

    docker build -t <your-rest-api-docker-image> .
    docker run -t <your-rest-api-docker-image> -d
    curl http://<DOCKER-IP>:5000/predict


# 3) Scalability and Real-time processing

Now imagine that we don't have just a few sensors but thousands of them constantly sending data to the cloud. We want to set up a data infrastructure that can handle this data. The architecture should contain a batch processing, as well as a real-time processing layer.

For instance, batch processing could be used to train (and regularly update) a predictive model for each of these devices. On the other hand, real-time processing could be used to apply simple data processing (e.g. aggregations over time, data cleaning etc.) or to use the trained models to the incoming data stream to do predictions.

How could such an architecture look like? Please provide a brief sketch.

Please, mention *anything* that might seem important to you, for instance:

- What kind of technologies would you use?
- Would you have a preferred Cloud infrastructure?
- Where/how would you store data? Multiple databases for different kinds of
  data? Which ones?
- On top of what is mentioned above, what would you use the batch/real-time
  pipeline for?
- What kind of questions would you need to have answered in order to solve this
  task more concretely or in order to make better decisions on architecture and
  technologies?

# Evaluation
Your challenge results will be evaluated based on the following criteria
- prediction accuracy (for the first part)
- code cleanliness
- level of creativity / ingenuity of your solution
