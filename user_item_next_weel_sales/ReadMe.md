# Task 1:
I would use Dirichlet distribution for probability of category purchase (because it'll be used as a prior later), Multinomial distribution for the probability of product choice within a
category, Poisson distribution for the probability for the purchased amount. I think external factors drive the category purchase incidence, like store location, weather, holidays etc. Product choice can be influenced by ads, position on the shelf, relative price etc
# Task 2:
### prod_prediction_roc_0.83.csv is the desired file
Run to get two prediction files, prod should be more accurate. Expected AUC is about 0.83 
```bash
python final_solution.py
```
This is not final solution, check Exploration.ipynb for other models. I stopped at boostings, but I can do a lot of feature engineering (i.e. purchases last week, relative user activity, item popularity etc), but it's too boring for the test task
