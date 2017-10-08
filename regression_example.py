import matplotlib
matplotlib.use('Agg')

# # Surrogate Modeling Example
# Imports
from functions import *

# ### Initialize the parameters and constants
# Set the ABM Evaluation Budget
budget = 500

# Set out-of-sample test and montecarlo sizes
test_size = 100
montecarlos = 100

# Set the Calibration Threshold
_CALIBRATION_THRESHOLD = 1.25

# Set the ABM parameters and support
islands_exploration_range = np.array([        
    (0.0,10), # rho
    (0.8,2.0), # alpha
    (0.0,1.0), # phi
    (0.0,1.0), # pi                                     
    (0.0,1.0), # eps
    (1,1000), # N
    (0.0,1.0)]) # Lambda

param_dims = islands_exploration_range.shape[0]

# # Surrogate Modeling as Regression Example
# This example illustrates the difference between Kriging, the (Batch) XGBoost algorithm and the (Iterative) XGBoost algorithm. The aim is to show the difference between Kriging and XGBoost on the exact same examples in the standard setting of regression. We do not expect a great difference in performance. 

# Evaluate entire Budget in Batch
n_dimensions = islands_exploration_range.shape[0]
evaluated_set_X_batch = get_sobol_samples(n_dimensions, budget, islands_exploration_range)
evaluated_set_y_batch = evaluate_islands_on_set(evaluated_set_X_batch)

# ## The critical issue with ABMs are the number of positive calibrations in the parameter space. Here, we select a high threshold to demonstrate the impact on the surrogate modeling performance.
print ("Number of positive calibrations available for training: ", (evaluated_set_y_batch>_CALIBRATION_THRESHOLD).sum())

# ## Kriging Surrogate
# We use the standard settings in the pyKriging package. This includes 10,000 evaluations of their selected
# optimizer, particle swarm optimization, for hyper-parameter estimation. We use a simple optimizer for our 
# hyper-parameters. For an equivalent comparison, the same optimizer and settings should be used. We do not 
# do this here because we depend on the pyKriging code and do not want to write our own kriging package. We
# also want to avoid using specialized hyper-parameter optimizers to demonstrate performance.
tic()
surrogate_models_kriging = kriging(evaluated_set_X_batch, evaluated_set_y_batch, random_state=0)
surrogate_models_kriging.train()
print ("Kriging Time: ", toc())

# ## XGBoost Surrogate
# This surrogate will not have multiple iterations. It will run on the entire budget of evaluations. Further, we use exactly the same optimizer and settings for Kriging and XGBoost.
tic()
surrogate_model_XGBoost = fit_surrogate_model(evaluated_set_X_batch,evaluated_set_y_batch)
print ("XGBoost Batch Time: ", toc())

# ## XGBoost Iterative Surrogate
tic()
surrogate_model_iterative, evaluated_set_X_iterative, evaluated_set_y_iterative = run_online_surrogate(budget, n_dimensions, islands_exploration_range, _CALIBRATION_THRESHOLD)
print ("XGBoost Iterative Time: ", toc())

def unique_rows(a):
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))

# Get an on out-of-sample test set that does not have combinations from the 
# batch or iterative experiments
final_test_size = (test_size*montecarlos)

# Generate unique test set
oos_set = get_sobol_samples(n_dimensions, final_test_size, islands_exploration_range)

selections = []
for i,v in enumerate(oos_set):
    if (v not in evaluated_set_X_batch) and (v not in evaluated_set_X_iterative):
        selections.append(i)
oos_set = oos_set[selections]

while oos_set.shape[0]<final_test_size:
    temp_samples = get_sobol_samples(n_dimensions, final_test_size, islands_exploration_range)
    
    selections = []
    for i,v in enumerate(temp_samples):
        if (v not in oos_set):
            selections.append(i)
    temp_samples = temp_samples[selections]
        
    oos_set = unique_rows(np.vstack([oos_set,temp_samples]))
          
oos_set = oos_set[:final_test_size]

# Evaluate the test set for the ABM response
tic()
y_test = evaluate_islands_on_set(oos_set)
print ("Time: ", toc())

# Evaluate both surrogates on the test set
y_hat_test = [None]*3
mse_perf = np.zeros((3,montecarlos))

tic()
y_hat_test[0] = np.array([surrogate_models_kriging.predict(v) for v in oos_set])
print ("Kriging Prediction Time: ", toc())

tic()
y_hat_test[1] = surrogate_model_XGBoost.predict(oos_set)
print ("XGBoost Prediction Time: ", toc())

tic()
y_hat_test[2] = surrogate_model_iterative.predict(oos_set)
print ("XGBoost (Iterative) Prediction Time: ", toc())

# MSE performance
for sur_idx in range(3):
    for i in range(montecarlos):
        mse_perf[sur_idx,i] = mean_squared_error(y_test[i*test_size:(i+1)*test_size],
                                                  y_hat_test[int(sur_idx)][i*test_size:(i+1)*test_size])
        
# ## Plot the densities for each of the methods
# Plot Performance Results
experiment_labels = ["Kriging","XGBoost (Batch)","XGBoost (Iterative)"]

mse_perf = pd.DataFrame(mse_perf,index=experiment_labels)
ax = plt.axes()
sns.heatmap(mse_perf, ax = ax)
ax.set_title('MSE Performance Heatmap over Monte-Carlos')
ax.set_xlabel('')

fig,ax = plt.subplots(figsize=(12, 5))

k_label = "Kriging: Mean " + str(mse_perf.iloc[0,:].mean()) + ", Variance " + str(mse_perf.iloc[0,:].var())
xgb_label = "XGBoost: Mean " + str(mse_perf.iloc[1,:].mean()) + ", Variance " + str(mse_perf.iloc[1,:].var())
xgbi_label = "XGBoost (iterative): Mean " + str(mse_perf.iloc[2,:].mean()) + ", Variance " + str(mse_perf.iloc[2,:].var())

fig1 = sns.distplot(mse_perf.iloc[0,:], label = k_label, ax=ax)
fig2 = sns.distplot(mse_perf.iloc[1,:], label = xgb_label, ax=ax)
fig3 = sns.distplot(mse_perf.iloc[2,:], label = xgbi_label, ax=ax)

plt.title("Out-Of-Sample Prediction Performance")
plt.xlabel('Mean-Squared Error')
plt.yticks(fig1.get_yticks(), fig1.get_yticks() / 100)
plt.ylabel('Density')
plt.legend()
fig.savefig("xgboost_kriging_ba_comparison_" + str(budget) + ".png");

experiment_labels.append("Islands")

sur_comparison = pd.DataFrame(np.vstack([y_hat_test,y_test]).T,columns=experiment_labels)

for exp in experiment_labels[:3]:
    colors = np.array(['b']*sur_comparison.shape[0])
    colors[(sur_comparison['Islands']<_CALIBRATION_THRESHOLD) & (sur_comparison[exp]>=_CALIBRATION_THRESHOLD)]='r'
    colors[(sur_comparison['Islands']>=_CALIBRATION_THRESHOLD) & (sur_comparison[exp]>=_CALIBRATION_THRESHOLD)]='g'
    colors[(sur_comparison['Islands']>=_CALIBRATION_THRESHOLD) & (sur_comparison[exp]<_CALIBRATION_THRESHOLD)]='y'

    fig,ax = plt.subplots(figsize=(6,6))
    
    plt.scatter(x=sur_comparison[exp],y=sur_comparison["Islands"],color=colors);
    
    plt.title("True Model vs Surrogate Model")
    plt.xlabel(exp, fontsize=16)
    plt.ylabel("Islands", fontsize=16)
    plt.xlim(-0.25,1.75)
    plt.ylim(-0.25,1.75)

    fig.savefig(exp + "_model_comparison_" + str(budget) + ".png");
