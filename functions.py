
""" Ignore Warnings """
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

""" Imports """
import numpy as np
import pandas as pd
import sobol_seq
from scipy.stats.distributions import entropy

""" surrogate models """
# Xtreeme Gradient Boosted Decision Trees
from xgboost import XGBRegressor, XGBClassifier

# Gaussian Process Regression (Kriging)
import pyKriging  
from pyKriging.krige import kriging  
from pyKriging.samplingplan import samplingplan

""" cross-validation
Cross validation is used in each of the rounds to approximate the selected 
surrogate model over the data samples that are available. 

The evaluated parameter combinations are randomly split into two sets. An 
in-sample set and an out-of-sample set. The surrogate is trained and its 
parameters are tuned to an in-sample set, while the out-of-sample performance 
is measured (using a selected performance metric) on the out-of-sample set. 
This out-of-sample performance is then used as a proxy for the performance 
on the full space of unevaluated parameter combinations. In the case of the 
proposed procedure, this full space is approximated by the randomly selected 
pool.
"""
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold

""" performance metric """
# Mean Squared Error
from sklearn.metrics import mean_squared_error, precision_score, median_absolute_error, f1_score

""" Algorithm Tuning Constants """
_N_EVALS = 200
_N_SPLITS = 10
_CALIBRATION_THRESHOLD = 1

# Functions
from time import time
def tic():
    global startTime_for_tictoc
    startTime_for_tictoc = time()

def toc():
    return str(time() - startTime_for_tictoc)

def evaluate_islands_on_set(parameter_combinations):
    """
    0,10), # rho
    (0.8,2), # alpha
    (0.0,1.0), # phi
    (0.0,1.0), # pi                                     
    (0.0,1.0), # eps
    (10,100), # N
    (0.0,1.0)]) # Lambda
    """
    y = np.zeros(parameter_combinations.shape[0])
    num_params = parameter_combinations.shape[1]
    
    if num_params == 1:
        for i,rho in enumerate(parameter_combinations):
            gdp = island_abm(rho=rho, 
                             _RNG_SEED=0)
            y[i] = calibration_measure(gdp)    
    elif num_params == 2:
        for i,(rho,alpha) in enumerate(parameter_combinations):
            gdp = island_abm(rho=rho, alpha=alpha,
                             _RNG_SEED=0)
            y[i] = calibration_measure(gdp)
        
    elif num_params == 3:
        for i,(rho,alpha,phi) in enumerate(parameter_combinations):
            gdp = island_abm(rho=rho, alpha=alpha, phi=phi, 
                             _RNG_SEED=0)
            y[i] = calibration_measure(gdp)
        
    elif num_params == 4:
        for i,(rho,alpha,phi,pi) in enumerate(parameter_combinations):
            gdp = island_abm(rho=rho, alpha=alpha, phi=phi, 
                             pi=pi, _RNG_SEED=0)
            y[i] = calibration_measure(gdp)
        
    elif num_params == 5:
        for i,(rho,alpha,phi,pi,eps) in enumerate(parameter_combinations):
            gdp = island_abm(rho=rho, alpha=alpha, phi=phi, 
                             pi=pi, eps=eps, _RNG_SEED=0)
            y[i] = calibration_measure(gdp)
        
    elif num_params == 6:
        for i,(rho,alpha,phi,pi,eps,N) in enumerate(parameter_combinations):
            gdp = island_abm(rho=rho, alpha=alpha, phi=phi, 
                             pi=pi, eps=eps, N=int(N),
                             _RNG_SEED=0)
            y[i] = calibration_measure(gdp)        
    elif num_params == 7:
        for i,(rho,alpha,phi,pi,eps,N,lambda_param) in enumerate(parameter_combinations):
            gdp = island_abm(rho=rho, alpha=alpha, phi=phi, 
                             pi=pi, eps=eps, N=int(N), 
                             lambda_param=lambda_param,
                             _RNG_SEED=0)
            y[i] = calibration_measure(gdp)
        
    return y

def island_abm(rho=0.01, 
               alpha=1.5, 
               phi=0.4, 
               pi=0.4, 
               eps=0.1, 
               lambda_param=1, 
               T=100, 
               N=50, 
               _RNG_SEED=0):
    """ Islands growth model

    Parameters
    ----------

    rho : 
    
    alpha : 
    
    phi : float, required
    
    eps : 
    
    lambda_param: (Default = 1)
        
    T : int, required
    The number of periods for the simulation
    
    N : int, optional (Default = 50)
    Number of firms
    
    _RNG_SEED : int, optional (Default = 0)
    Random number seen


    Output
    ------

    GDP : array, length = [,T]
    Simulated GPD

    """
    # Set random number seed
    np.random.seed(_RNG_SEED) 

    T_2 = int(T/2)
    
    GDP = np.zeros((T, 1))

    # Distributions
    # Precompute random binomial draws
    xy = np.random.binomial(1, pi, (T, T)) 
    xy[T_2, T_2] = 1

    # Containers
    s = np.zeros((T, T))
    A = np.ones((N, 6))

    # Initializations
    A[:, 1] = T_2
    A[:, 2] = T_2
    m = np.zeros((T, T))
    m[T_2, T_2] = N
    dest = np.zeros((N, 2))
    
    """ Begin ABM Code """
    for t in range(T):
        w = np.zeros((N, N))
        signal = np.zeros((N, N))

        for i in range(N):
            for j in range(N):
                if i != j:
                    if A[j, 0] == 1:
                        w[i, j] = np.exp(-rho*(np.abs(A[j,1]-A[i,1])+ \
                                               np.abs(A[j,2]-A[i,2])))

                        if np.random.rand() < w[i,j]:
                            signal[i,j]=s[int(A[j, 1]), int(A[j, 2])]

            if A[i,0] == 1:
                A[i,4] = s[int(A[i,1]),int(A[i,2])] * \
                            m[int(A[i,1]),int(A[i,2])]**alpha
                A[i,3] = s[int(A[i, 1]), int(A[i, 2])]

            if A[i, 0] == 3:
                A[i, 4] = 0
                rnd = np.random.rand()
                if rnd <= 0.25:
                    A[i,1] += 1
                else:
                    if rnd <= 0.5:
                        A[i,1] -= 1
                    else:
                        if rnd <= 0.75:
                            A[i,2] += 1
                        else:
                            A[i,2] -= 1

                if xy[int(A[i,1]),int(A[i,2])] == 1:
                    A[i,0] = 1
                    m[int(A[i,1]),int(A[i,2])] += 1
                    if m[int(A[i,1]),int(A[i,2])] == 1:
                        s[int(A[i,1]),int(A[i,2])] = \
                            (1+int(np.random.poisson(lambda_param)))* \
                            (A[i,1]+A[i,2])+phi*A[i,5]+np.random.randn()

            if (A[i,0] == 1) and (np.random.rand() <= eps):
                A[i,0] = 3
                A[i,5] = A[i,4]
                m[int(A[i,1]),int(A[i,2])] -= 1

            if t > T/100:
                if A[i,0] == 2:
                    A[i,4] = 0
                    if dest[i,0] != A[i,1]:
                        if dest[i,0] > A[i,1]:
                            A[i,1] += 1
                        else:
                            A[i,1] -= 1
                    else:
                        if dest[i,1] != A[i,2]:
                            if dest[i,1] > A[i,2]:
                                A[i,2] += 1
                            else:
                                A[i,2] -= 1
                    if (dest[i,0] == A[i,1]) and (dest[i,1] == A[i,2]):
                        A[i,0] = 1
                        m[int(dest[i,0]),int(dest[i,1])] += 1
                if A[i,0] == 1:
                    best_sig = np.max(signal[i,:])
                    if best_sig > s[int(A[i,1]),int(A[i,2])]:
                        A[i,0] = 2
                        A[i,5] = A[i,4]
                        m[int(A[i,1]),int(A[i,2])] -= 1
                        index = np.where(signal[i,:] == best_sig)[0]
                        if index.shape[0] > 1:
                            ind = int(index[int(np.random.uniform(0, len(index)))])
                        else:
                            ind = int(index)
                        dest[i,0] = A[ind,1]
                        dest[i,1] = A[ind,2]

        GDP[t, 0] = np.sum(A[:, 4])
    
    log_GDP = np.log(GDP)
    
    return log_GDP

def calibration_measure(log_GDP):
    """ Calibration Measure
    
    Input
    -----
    
    GDP : array, required, length = [,T]
    
    Output
    ------
    
    agdp : float
    Average GDP growth rate
    
    """
    T = log_GDP.shape[0]
    log_GDP = log_GDP[~np.isinf(log_GDP)]
    log_GDP = log_GDP[~np.isnan(log_GDP)]
    if log_GDP.shape[0]>0:
        GDP_growth_rate = (log_GDP[-1] - log_GDP[0])/T
    else:
        GDP_growth_rate = 0

    return GDP_growth_rate

def calibration_condition(average_GDP_growth_rate,threshold_condition):
    return average_GDP_growth_rate>=threshold_condition

def set_surrogate_as_gbt():
    """ Set the surrogate model as Gradient Boosted Decision Trees
    Helper function to set the surrogate model and parameter space 
    as Gradient Boosted Decision Trees.
    
    For detail, see: 
    http://scikit-learn.org/stable/modules/generated/
    sklearn.ensemble.GradientBoostingRegressor.html
    
    
    Parameters
    ----------
    
    None
    
    Returns
    -------
    
    surrogate_model : 
    
    surrogate_parameter_space : 
    
    
    """
    
    surrogate_model = XGBRegressor(seed=0)

    surrogate_parameter_space = [
        (25, 100), # n_estimators
        (0.001, 1), # learning_rate
        (1, 100), # max_depth
        (0.0, 1), # reg_alpha
        (0.0, 1), # reg_lambda
        (0.5, 1.0)] # subsample
        
    return surrogate_model, surrogate_parameter_space

def set_surrogate_as_gpr():
    """ Set the surrogate model as Gaussian Process Regression (Kriging)
    Helper function to set the surrogate model and parameter space for
    Gaussian Process Regression (Kriging)
    
    For detail, see: 
    http://scikit-learn.org/stable/modules/generated/
    sklearn.gaussian_process.GaussianProcessRegressor.html
    
    Parameters are optimized using L-BFGS from the data and a Radial Basis 
    Function is used as the Kernel.
    
    Parameters
    ----------
    
    None# Set the surrogaet parameter settings to the output of surrogate_model_tuned
    
    Returns
    -------
    
    surrogate_model : 
    
    surrogate_parameter_space : 
    
    
    """
    
    surrogate_model = GaussianProcessRegressor(random_state=0)

    surrogate_parameter_space  = []
    
    return surrogate_model, surrogate_parameter_space

def custom_metric_regression(y_hat, y):
#     return 'MSE', median_absolute_error(y.get_label(), y_hat)
    return 'MSE', mean_squared_error(y.get_label(), y_hat)

def custom_metric_binary(y_hat, y):
    return 'MSE', f1_score(y.get_label(), y_hat, average='weighted')

def fit_surrogate_model(X,y, surrogate_model, surrogate_parameter_space):
    """ Fit a surrogate model to the X,y parameter combinations
    
    Parameters
    ----------
    
    surrogate_model : 
    
    X : 
    
    y : 
    
    
    Output
    ------
    surrogate_model_fitted : A surrogate model fitted 
    
    """
    def objective(params):
        n_estimators, learning_rate, max_depth, reg_alpha, \
        reg_lambda, subsample = params

        reg = XGBRegressor( n_estimators=n_estimators,
                            learning_rate=learning_rate,
                            max_depth=max_depth,
                            reg_alpha=reg_alpha,
                            reg_lambda=reg_lambda,
                            subsample=subsample,
                            seed=0)
        
        kf = KFold(n_splits=_N_SPLITS,random_state=0,shuffle=True)
        kf_cv = [(train,test) for train,test in kf.split(X,y)] 
            
        return -np.max(cross_val_score(reg, 
                                        X, y, 
                                        cv=kf_cv, 
                                        n_jobs=1,
                                        fit_params={'eval_metric':custom_metric_regression},
                                        scoring="neg_median_absolute_error"))

    from skopt import gp_minimize    

    # We assume the surrogate model parameter space is smooth, so we 
    # use Gaussian Process Regression to optimize the Hyper-Parameters.
    surrogate_model_tuned = gp_minimize(objective, 
                                        surrogate_parameter_space, 
                                        n_calls=_N_EVALS,
                                        n_restarts_optimizer=100,
                                        acq_func="LCB",
                                        n_jobs=-1,
                                        random_state=0)

    surrogate_model.set_params(n_estimators=surrogate_model_tuned.x[0],
                               learning_rate=surrogate_model_tuned.x[1],
                               max_depth=surrogate_model_tuned.x[2],
                               reg_alpha=surrogate_model_tuned.x[3],
                               reg_lambda=surrogate_model_tuned.x[4],
                               subsample=surrogate_model_tuned.x[5],
                               seed=0)

    surrogate_model.fit(X,y,eval_metric=custom_metric_regression)
    
    return surrogate_model

def fit_entropy_classifier(X,y,calibration_threshold):
    """ Fit a surrogate model to the X,y parameter combinations
    
    Parameters
    ----------
    
    surrogate_model : 
    
    X : 
    
    y : 
    
    
    Output
    ------
    surrogate_model_fitted : A surrogate model fitted 
    
    """
    y_binary = calibration_condition(y,calibration_threshold)
    _, surrogate_parameter_space = set_surrogate_as_gbt()
    
    def objective(params):
        n_estimators, learning_rate, max_depth, reg_alpha, \
        reg_lambda, subsample = params

        clf = XGBClassifier(n_estimators=n_estimators,
                            learning_rate=learning_rate,
                            max_depth=max_depth,
                            reg_alpha=reg_alpha,
                            reg_lambda=reg_lambda,
                            subsample=subsample,
                            seed=0,
                            objective="binary:logistic")
                
        skf = StratifiedKFold(n_splits=_N_SPLITS,random_state=0,shuffle=True)
        skf_cv = [(train,test) for train,test in skf.split(X,y_binary)]        

        return -np.max(cross_val_score(clf, 
                                        X, y_binary, 
                                        cv=skf_cv, 
                                        n_jobs=1,
                                        fit_params={'eval_metric':custom_metric_binary},
                                        scoring="f1_weighted"))

    from skopt import gp_minimize    

    # We assume the surrogate model parameter space is smooth, so we 
    # use Gaussian Process Regression to optimize the Hyper-Parameters.
    clf_tuned = gp_minimize(objective, 
                            surrogate_parameter_space, 
                            n_calls=_N_EVALS,
                            n_restarts_optimizer=100,
                            acq_func="LCB",
                            n_jobs=-1, 
                            random_state=0)

    clf = XGBClassifier(n_estimators=clf_tuned.x[0],
                        learning_rate=clf_tuned.x[1],
                        max_depth=clf_tuned.x[2],
                        reg_alpha=clf_tuned.x[3],
                        reg_lambda=clf_tuned.x[4],
                        subsample=clf_tuned.x[5],
                        seed=0)
    
    clf.fit(X,y_binary,eval_metric=custom_metric_binary)
    
    return clf

def run_online_surrogate(budget, n_dimensions, islands_exploration_range, calibration_threshold):
    print "XGBoost Started!"
    # 1. Draw the Pool
    # Set pool size
    pool_size = 1000000

    # Draw Pool
    pool = get_sobol_samples(n_dimensions, pool_size, islands_exploration_range)

    # 2. Randomly Draw and Evaluate Initialization Set
    # Draw the initialization set from the pool as a permutation of the pool index.

    # Set number of selections per round
    samples_to_select = np.ceil(np.log(pool_size)).astype(int)

    # Set initialization samples
    initialization_samples = np.random.permutation(pool_size)[:samples_to_select]

    # Evaluate Initialization Set
    evaluated_set_X = pool[initialization_samples]
    evaluated_set_y = evaluate_islands_on_set(evaluated_set_X)

    # Update unevaluated_set_X
    unevaluated_set_X = pool[list(set(range(pool_size)) - set(initialization_samples))]

    surrogate_model, surrogate_parameter_space = set_surrogate_as_gbt()
    print "Evaluated set size: ", evaluated_set_y.shape[0]

    while evaluated_set_y.shape[0] < budget:
        # 3. Build Surrogate on evaluated samples
        surrogate_model_this_round = fit_surrogate_model( \
            evaluated_set_X,evaluated_set_y,
            surrogate_model=surrogate_model,
            surrogate_parameter_space=surrogate_parameter_space)

        # 4. Predict Response over Pool
        predict_response_pool = surrogate_model_this_round.predict(unevaluated_set_X)
        predicted_positives = calibration_condition(predict_response_pool,
                                                    calibration_threshold)
        num_predicted_positives = predicted_positives.sum()

        # 5. Select small subset of Pool for Evaluation
        evaluated_set_X, evaluated_set_y, unevaluated_set_X = get_round_selections( \
                evaluated_set_X,evaluated_set_y,
                unevaluated_set_X,
                predicted_positives, num_predicted_positives,
                samples_to_select, calibration_threshold,
                budget)

    # 6. Output Final Surrogate Model
    surrogate_model = fit_surrogate_model(evaluated_set_X,evaluated_set_y,
                               surrogate_model=surrogate_model,
                               surrogate_parameter_space=surrogate_parameter_space)

    return surrogate_model, evaluated_set_X, evaluated_set_y

def get_sobol_samples(n_dimensions, samples, parameter_support):
    """
    
    
    """
    # Get the range for the support
    support_range = parameter_support[:,1] - parameter_support[:,0]
    
    # Generate the Sobol samples
    random_samples = sobol_seq.i4_sobol_generate(n_dimensions, samples)
    
    # Compute the parameter mappings between the Sobol samples and supports
    sobol_samples = np.vstack([
        np.multiply(s,support_range)+parameter_support[:,0] 
               for s in random_samples])
    
    return sobol_samples

def get_round_selections(evaluated_set_X,evaluated_set_y,
                         unevaluated_set_X,
                         predicted_positives, num_predicted_positives,
                         samples_to_select, calibration_threshold,
                         budget):
    """
    
    
    """
    samples_to_select = np.min([abs(budget - evaluated_set_y.shape[0]),
                                samples_to_select]).astype(int)
    
    if num_predicted_positives>=samples_to_select:
        round_selections = int(samples_to_select)
        selections = np.where(predicted_positives==True)[0]
        selections = np.random.permutation(selections)[:round_selections]

    elif num_predicted_positives<=samples_to_select:
        # select all predicted positives
        selections = np.where(predicted_positives==True)[0]

        # select remainder according to entropy weighting
        budget_shortfall = int(samples_to_select - num_predicted_positives)
        
        selections = np.append(selections,
                               get_new_labels_entropy(evaluated_set_X, evaluated_set_y,
                                                      unevaluated_set_X, 
                                                      calibration_threshold,
                                                      budget_shortfall))

    else: # if we don't have any predicted positive calibrations
        selections = get_new_labels_entropy(clf, unevaluated_set_X, samples_to_select)

    to_be_evaluated = unevaluated_set_X[selections]
    unevaluated_set_X = np.delete(unevaluated_set_X,selections,0)
    evaluated_set_X = np.vstack([evaluated_set_X, to_be_evaluated])
    evaluated_set_y = np.append(evaluated_set_y,evaluate_islands_on_set(to_be_evaluated))
    
    return evaluated_set_X, evaluated_set_y, unevaluated_set_X

def get_new_labels_entropy(evaluated_set_X, evaluated_set_y,
                           unevaluated_X, calibration_threshold,
                           number_of_new_labels):
    """ Get a set of parameter combinations according to their predicted label entropy
    
    
    
    """    
#     if _KRIGING:
#         clf = GaussianProcessClassifier()
#         clf.fit(evaluated_set_X,calibration_condition(evaluated_set_y,calibration_threshold))
#     else:
    clf = fit_entropy_classifier(evaluated_set_X,evaluated_set_y,calibration_threshold)

    y_hat_probability = clf.predict_proba(unevaluated_X)
    y_hat_entropy = np.array(map(entropy,y_hat_probability))
    y_hat_entropy /= y_hat_entropy.sum()
    unevaluated_X_size = unevaluated_X.shape[0]
        
    selections = np.random.choice(a = unevaluated_X_size,
                                  size = number_of_new_labels,
                                  replace = False,
                                  p = y_hat_entropy)    
    return selections