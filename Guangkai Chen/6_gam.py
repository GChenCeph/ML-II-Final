import json
import numpy as np
import pandas as pd
from pygam import LinearGAM, s
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

# Load data
with open('hu_hotel_dropped.json', 'r') as file:
    hotel_revenue = json.load(file)

# Split data into training and testing sets
train_data = hotel_revenue[:-50]
test_data = hotel_revenue[-50:]

#################################################################
####################### Training Phase ##########################

# Extract predictors and target from training data
ranks = np.array([hotel['Rank'] for hotel in train_data])
scores = np.array([hotel['Total Score'] for hotel in train_data])
avg = np.array([hotel['avg_distance'] for hotel in train_data])
spearman = np.array([hotel['spearman_weighted_avg_distance'] for hotel in train_data])
pearson = np.array([hotel['pearson_weighted_avg_distance'] for hotel in train_data])
revenue = np.array([hotel['revenue'] for hotel in train_data])
vector = np.array([hotel['distance_vector'] for hotel in train_data])
size = len(ranks)

#print(revenue)

# Phase 1: rank and score
log_avg = np.log(avg + 1)
log_spearman = np.log(spearman + 1)
log_pearson = np.log(pearson + 1)
log_vector = np.log(vector + 1)

# Combine transformed predictors
predictor = np.column_stack([log_avg, log_spearman, log_pearson, log_vector])
#predictor = np.column_stack([avg, spearman, pearson, vector])

lambdas = np.logspace(-3, 3, 11)

n_splines = 25
spline_order = 3

gam_rank = LinearGAM(s(0, n_splines=n_splines, spline_order=spline_order) +
                     s(1, n_splines=n_splines, spline_order=spline_order) +
                     s(2, n_splines=n_splines, spline_order=spline_order) +
                     s(3, n_splines=n_splines, spline_order=spline_order))

gam_score = LinearGAM(s(0, n_splines=n_splines, spline_order=spline_order) +
                      s(1, n_splines=n_splines, spline_order=spline_order) +
                      s(2, n_splines=n_splines, spline_order=spline_order) +
                      s(3, n_splines=n_splines, spline_order=spline_order))

# Define a range of lambda values for cross-validation
lambdas = np.logspace(-3, 3, 11)

# Perform grid search for the rank model
best_gam_rank = gam_rank.gridsearch(predictor, ranks, lam=lambdas)

# Perform grid search for the score model
best_gam_score = gam_score.gridsearch(predictor, scores, lam=lambdas)

# Predictions for rank and score using the best models
predicted_ranks = best_gam_rank.predict(predictor)
predicted_scores = best_gam_score.predict(predictor)

# Calculate residuals
residuals_rank = np.abs(ranks - predicted_ranks) #np.abs
residuals_score = np.abs(scores - predicted_scores)

print("residuals_rank : ", sum(residuals_rank) / size) #residuals_rank, 
print("residuals_score : ", sum(residuals_score) / size) #residuals_score, 

# Phase 2: revenue prediction
# Step 1: Individual predictors
log_revenue = np.log(revenue + 1)

predictors_names = ["ranks", "scores", "avg", "p_avg", "s_avg", "d_v"]
revenue_predictor = np.column_stack([predicted_ranks, predicted_scores, log_avg, log_spearman, log_pearson, log_vector])

models = {}
revenue_prediction = {}

def gam_pred(predictors_names, predictors, target, n_splines, spline_order, lambdas):

    for i, name in enumerate(predictors_names):
        
        predictor_2d = predictors[:, i].reshape(-1, 1)

        gam_model = LinearGAM(s(0, n_splines=n_splines, spline_order=spline_order))
        best_gam_model = gam_model.gridsearch(predictor_2d, target, lam=lambdas)

        models[name] = best_gam_model
        predicted = best_gam_model.predict(predictor_2d)
        revenue_prediction[name] = predicted
        print("Residuals Average for {}: ".format(name), np.mean(np.abs(target - predicted)))

gam_pred(predictors_names, revenue_predictor, log_revenue, n_splines, spline_order, lambdas)

# step 2: all-in
gam_revenue = LinearGAM(s(0, n_splines=n_splines, spline_order=spline_order) +
                        s(1, n_splines=n_splines, spline_order=spline_order) +
                        s(2, n_splines=n_splines, spline_order=spline_order) +
                        s(3, n_splines=n_splines, spline_order=spline_order) +
                        s(4, n_splines=n_splines, spline_order=spline_order) +
                        s(5, n_splines=n_splines, spline_order=spline_order))

# Perform grid search with the correct model
best_gam_revenue = gam_revenue.gridsearch(revenue_predictor, log_revenue, lam=lambdas)

# Prediction using the correct model
revenued = best_gam_revenue.predict(revenue_predictor)
revenue_prediction["all"] = revenued

print("All-in Residuals Average: ", np.mean(np.abs(log_revenue - revenued)))

#################################################################
######################## Testing Phase ##########################

# Extract predictors and target from training data
test_ranks = np.array([hotel['Rank'] for hotel in test_data])
test_scores = np.array([hotel['Total Score'] for hotel in test_data])
test_avg = np.array([hotel['avg_distance'] for hotel in test_data])
test_spearman = np.array([hotel['spearman_weighted_avg_distance'] for hotel in test_data])
test_pearson = np.array([hotel['pearson_weighted_avg_distance'] for hotel in test_data])
test_revenue = np.array([hotel['revenue'] for hotel in test_data])
test_vector = np.array([hotel['distance_vector'] for hotel in test_data])
test_size = len(ranks)

# Phase 1: rank and score
log_test_avg = np.log(test_avg + 1)
log_test_spearman = np.log(test_spearman + 1)
log_test_pearson = np.log(test_pearson + 1)
log_test_vector = np.log(test_vector + 1)

test_predictor = np.column_stack([log_test_avg, log_test_spearman, log_test_pearson, log_test_vector])

predicted_test_ranks = best_gam_rank.predict(test_predictor)
predicted_test_scores = best_gam_score.predict(test_predictor)

test_residuals_rank = np.abs(test_ranks - predicted_test_ranks)
test_residuals_score = np.abs(test_scores - predicted_test_scores)

print("predicted_test_ranks: ", sum(test_residuals_rank) / size)
print("predicted_test_scores: ", sum(test_residuals_score) / size)

log_test_revenue = np.log(test_revenue + 1)

#print(log_test_revenue)

test_predictor_full = np.column_stack([predicted_test_ranks, predicted_test_scores, log_test_avg, log_test_spearman, log_test_pearson, log_test_vector])

for i, (name, model) in enumerate(models.items()):

    test_predictor_single = test_predictor_full[:, i]

    predicted = model.predict(test_predictor_single)
    print("Test Residuals Average for {}: ".format(name), np.mean(np.abs(log_test_revenue - predicted)))

predicted_all = best_gam_revenue.predict(test_predictor_full)

#print("Test All-in Residuals: ", log_test_revenue - predicted_all)
print("Test All-in Residuals Average: ", np.mean(np.abs(log_test_revenue - predicted_all)))