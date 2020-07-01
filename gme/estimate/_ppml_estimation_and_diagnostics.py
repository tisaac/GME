__author__ = "Saad Ahmad"
__project__ = "gme.estimate"
__created__ = "05-24-2018"
__edited__ = "Peter Herman"

import pandas as pd
import numpy as np
from typing import List
from typing import Union
import statsmodels.api as sm
import time as time
import traceback
import scipy

#-----------------------------------------------------------------------------------------#
# This file contains the underlying functions for the .estimate method in EstimationModel #
#-----------------------------------------------------------------------------------------#

# -----------
# Main function: Sequences the other routines
# -----------

def _estimate_ppml(data_frame,
                   meta_data,
                   specification,
                   fixed_effects: List[Union[str,List[str]]] = [],
                   drop_fixed_effect: List[Union[str,List[str]]] = []):
    '''
    Performs sector by sector GLM estimation with PPML diagnostics

    Args:
        data_frame: (Pandas.DataFrame) A dataframe containing data for estimation
        meta_data: (obj) a MetaData object from gme.EstimationData
        specification: (obj) a Specification object from gme.EstimationModel
        fixed_effects: (List[Union[str,List[str]]]) A list of variables to construct fixed effects based on.
            Can accept single string entries, which create fixed effects corresponding to that variable or lists of
            strings that create fixed effects corresponding to the interaction of the list items. For example,
            fixed_effects = ['importer',['exporter','year']] would create a set of importer fixed effects and a set of
            exporter-year fixed effects.
        drop_fixed_effect: (List[Union[str,List[str]]]) The fixed effect category from which to drop a fixed effect.
            The entry should be a subset of the list supplied for fixed_effects. In each case, the last fixed effect is
            dropped.

    Returns: (Dict[GLM.fit], Pandas.DataFrame, Dict[DataFrame])
        1. Dictionary of statsmodels.GLM.fit objects with sectors as the keys.
        2. Dataframe with diagnostic information by sector
        3. Dictionary of estimation DataFrames + predicted trade values with sectors as the keys.
    '''

    post_diagnostics_data_frame_dict = {}
    results_dict = {}
    diagnostics_log = pd.DataFrame([])
    start_time = time.time()
    print('Estimation began at ' + time.strftime('%I:%M %p  on %b %d, %Y'))

    if not specification.sector_by_sector:
        data_frame = data_frame.reset_index(drop=True)
        fixed_effects_df = _generate_fixed_effects(data_frame, fixed_effects, drop_fixed_effect)
        estimating_data_frame = pd.concat(
            [data_frame[specification.lhs_var], data_frame[specification.rhs_var], fixed_effects_df], axis=1)
        model_fit, post_diagnostics_data_frame, diagnostics_log = _regress_ppml(estimating_data_frame, specification)
        results_dict['all'] = model_fit
        end_time = time.time()
        diagnostics_log.at['Completion Time'] = str(round((end_time - start_time)/60,2)) + ' minutes'
        post_diagnostics_data_frame_dict['all'] = post_diagnostics_data_frame


    else:
        sector_groups = data_frame.groupby(meta_data.sector_var_name)
        sector_list = _sectors(data_frame, meta_data)
        iteration_count = 1
        for sector in sector_list:
            sector_start_time = time.time()
            print('Sector ' + str(sector) + ' began at ' + time.strftime('%I:%M %p  on %b %d, %Y'))
            sector_data_frame = sector_groups.get_group(sector)
            sector_data_frame = sector_data_frame.reset_index(drop=True)

            # Create fixed effects
            fixed_effects_df = _generate_fixed_effects(sector_data_frame, fixed_effects, drop_fixed_effect)

            # Dataframe for estimations
            sector_data_frame = pd.concat(
                [sector_data_frame[specification.lhs_var], sector_data_frame[specification.rhs_var], fixed_effects_df],
                axis=1)
            model_fit, post_diagnostics_data_frame, diagnostics_output = _regress_ppml(sector_data_frame, specification)

            # Timing reports
            sector_end_time = time.time()
            diagnostics_output.at['Sector Completion Time'] = (str(round((sector_end_time - sector_start_time) / 60, 2))
                                                               + ' minutes')
            if iteration_count > 1:
                average_time = ((time.time() - start_time) / 60) / iteration_count
                completion_time = (len(sector_list) - iteration_count) * average_time
                print("Average iteration time:  " + str(average_time) + "  minutes")
                print("Expected time to completion:  " + str(completion_time) + " minutes ("+ str(completion_time/60)
                      + " hours)\n")

            # Store results
            post_diagnostics_data_frame_dict[str(sector)] = post_diagnostics_data_frame
            results_dict[str(sector)] = model_fit
            diagnostics_log = pd.concat([diagnostics_log, diagnostics_output.rename(str(sector))], axis=1)
            iteration_count+=1

    print("Estimation completed at " + time.strftime('%I:%M %p  on %b %d, %Y'))
    return results_dict, diagnostics_log, post_diagnostics_data_frame_dict


# --------------
# Prep for Estimation Functions
# --------------


def _generate_fixed_effects(data_frame,
                            fixed_effects: List[Union[str,List[str]]] = [],
                            drop_fixed_effect: List[Union[str,List[str]]] = []):
    '''
    Create fixed effects for single and interacted categorical variables.

    Args:
        data_frame: Pandas.DataFrame
            A DataFrame containing data for estimation.
        fixed_effects: List[Union[str,List[str]]]
            A list of variables to construct fixed effects based on.
            Can accept single string entries, which create fixed effects corresponding to that variable or lists of
            strings, which create fixed effects corresponding to the interaction of the list items. For example,
            fixed_effects = ['importer',['exporter','year']] would create a set of importer fixed effects and a set of
            exporter-year fixed effects.
        drop_fixed_effect: (optional) List[Union[str,List[str]]]
            The fixed effect category from which to drop a fixed effect.
            The entry should be a subset of the list supplied for fixed_effects. In each case, the last fixed effect
            is dropped.
    Returns: Pandas.DataFrame
        A DataFrame of fixed effects to be concatenated with the estimating DataFrame
    '''

    fixed_effect_data_frame = pd.DataFrame([])
    # Get list for separate and combine fixed effect
    combined_fixed_effects = []
    separate_fixed_effects = []

    for item in fixed_effects:
        if type(item) is list:
            combined_fixed_effects.append(item)
        else:
            separate_fixed_effects.append(item)
    # Construct simple fixed effects
    for category in separate_fixed_effects:
        name = category + '_fe'
        temp_fe = pd.get_dummies(data_frame[category], prefix=name)

        if category in drop_fixed_effect:
            temp_fe.drop(temp_fe.columns[[temp_fe.shape[1] - 1]], axis=1, inplace=True)

        fixed_effect_data_frame = pd.concat((fixed_effect_data_frame, temp_fe), axis=1)
    # Construct multiple fixed effects
    for item in combined_fixed_effects:
        if len(item) < 1:
            raise ValueError('A fixed_effects list element cannot be an empty list [].')

        if len(item) == 1:
            name = '_'.join(item) + '_fe'
            temp_fe = pd.get_dummies(data_frame[item[0]], prefix=name)

            if item in drop_fixed_effect:
                temp_fe.drop(temp_fe.columns[[temp_fe.shape[1] - 1]], axis=1, inplace=True)

            fixed_effect_data_frame = pd.concat((fixed_effect_data_frame, temp_fe), axis=1)

        elif len(item) > 1:
            name = '_'.join(item) + '_fe'
            temp_data_frame = data_frame.loc[:,item]
            temp_data_frame.loc[:,name] = temp_data_frame.astype(str).sum(axis=1).copy()
            temp_fe = pd.get_dummies(temp_data_frame[name], prefix=name)

            if item in drop_fixed_effect:
                temp_fe.drop(temp_fe.columns[[temp_fe.shape[1] - 1]], axis=1, inplace=True)

            fixed_effect_data_frame = pd.concat((fixed_effect_data_frame, temp_fe), axis=1)

    fixed_effect_data_frame = fixed_effect_data_frame.reset_index(drop=True)

    return fixed_effect_data_frame


def _sectors(data_frame, meta_data):
    '''
    A function to extract a list of sectors from the estimating data_frame
    :param data_frame: (Pandas.DataFrame) A pandas data frmae to be used for estimation with a column defining
    sector or product IDs.
    :param meta_data: (obj) a MetaData object from gme.EstimationData
    :return: (list) a list of sorted sector IDs.
    '''
    sector_list = data_frame[meta_data.sector_var_name].unique()
    sector_list = np.ndarray.tolist(sector_list)
    sector_list.sort()
    return sector_list


# -------------
# PPML Regression and pre-diagnostics
# -------------

from statsmodels.tools.validation import float_like

class _SparseGLM(sm.GLM):
    '''
    This class keeps a sparse copy of the exogenous variables, `self.spexog`,
    and uses `self.spexog.dot(x)` in place of `np.dot(self.exog, x)`.
    '''
    def __init__(self, *args, **kwargs):
        super(_SparseGLM, self).__init__(*args, **kwargs)
        self.spexog = scipy.sparse.csr_matrix(self.exog)

    def loglike(self, params, scale=None):
        """the log-likelihood for a generalized linear model
        """
        scale = float_like(scale, "scale", optional=True)
        # replaced np.dot(self.exog, params) here
        lin_pred = self.spexog.dot(params) + self._offset_exposure
        expval = self.family.link.inverse(lin_pred)
        if scale is None:
            scale = self.estimate_scale(expval)
        llf = self.family.loglike(self.endog, expval, self.var_weights,
                                  self.freq_weights, scale)
        return llf

    def score(self, params, scale=None):
        """score, first derivative of the loglikelihood function
        """
        scale = float_like(scale, "scale", optional=True)
        score_factor = self.score_factor(params, scale=scale)
        # replaced np.dot(score_factor, self.exog)
        return self.spexog.T.dot(score_factor)

    def hessian(self, params, scale=None, observed=None):
        """Hessian, second derivative of loglikelihood function
        """
        if observed is None:
            if getattr(self, '_optim_hessian', None) == 'eim':
                observed = False
            else:
                observed = True
        scale = float_like(scale, "scale", optional=True)

        factor = self.hessian_factor(params, scale=scale, observed=observed)
        factord = scipy.sparse.diags([factor],[0])
        return (-self.spexog.T.dot(factord.dot(self.spexog))).toarray()

    def predict(self, params, exog=None, exposure=None, offset=None,
                linear=False):
        """predicted values for a design matrix
        """

        # Use fit offset if appropriate
        if offset is None and exog is None and hasattr(self, 'offset'):
            offset = self.offset
        elif offset is None:
            offset = 0.

        if exposure is not None and not isinstance(self.family.link,
                                                   families.links.Log):
            raise ValueError("exposure can only be used with the log link "
                             "function")

        # Use fit exposure if appropriate
        if exposure is None and exog is None and hasattr(self, 'exposure'):
            # Already logged
            exposure = self.exposure
        elif exposure is None:
            exposure = 0.
        else:
            exposure = np.log(exposure)

        if exog is None:
            exog = self.spexog

        # replace linpred = np.dot(exog, params) + offset + exposure
        linpred = exog.dot(params) + offset + exposure
        if linear:
            return linpred
        else:
            return self.family.fitted(linpred)


def _fit_newton_line_search(f, score, start_params, fargs, kwargs, disp=True,
                            maxiter=100, callback=None, retall=False,
                            full_output=True, hess=None):
    '''
    Newton's method like `_fit_newton` from `statsmodels`, but with
    globalization on the objective using `scipy.optimize.line_search`.
    '''
    tol = kwargs.setdefault('tol', 1e-8)
    iterations = 0
    oldparams = np.inf
    newparams = np.asarray(start_params)
    if retall:
        history = [oldparams, newparams]
    obj = f(newparams)
    while iterations < maxiter:
        g = score(newparams)
        H = np.asarray(hess(newparams))
        delta = np.linalg.solve(H,-g)
        alpha, _, _, obj, _, _ = scipy.optimize.line_search(f, score,
                                                            newparams, delta,
                                                            g, obj)
        if alpha is None:
            # line search break down, probably due to numerically zero
            # delta.dot(g), end optimization
            break
        delta *= alpha
        newparams = newparams + delta
        if retall:
            history.append(newparams)
        if callback is not None:
            callback(newparams)
        if np.max(np.abs(delta)) < tol:
            break
        iterations +=1
    fval = f(newparams, *fargs)  # this is the negative log-likelihood
    if iterations == maxiter:
        warnflag = 1
        if disp:
            print("Warning: Maximum number of iterations has been "
                   "exceeded.")
            print("         Current function value: %f" % fval)
            print("         Iterations: %d" % iterations)
    else:
        warnflag = 0
        if disp:
            print("Optimization terminated successfully.")
            print("         Current function value: %f" % fval)
            print("         Iterations %d" % iterations)
    if full_output:
        (xopt, fopt, niter,
         gopt, hopt) = (newparams, f(newparams, *fargs),
                        iterations, score(newparams),
                        hess(newparams))
        converged = not warnflag
        retvals = {'fopt': fopt, 'iterations': niter, 'score': gopt,
                   'Hessian': hopt, 'warnflag': warnflag,
                   'converged': converged}
        if retall:
            retvals.update({'allvecs': history})

    else:
        xopt = newparams
        retvals = None

    return xopt, retvals


def _regress_ppml(data_frame, specification):
    '''
    Perform a GLM estimation with collinearity, insufficient variation, and overfit diagnostics and corrections.
    :param data_frame: (Pandas.DataFrame) A DataFrame for estimation
    :param specification: (obj) a Specification object from gme.EstimationModel
    :return: (GLM.fit() obj, Pandas.DataFrame, Pandas.Series)
        1. The first returned object is a GLM.fit() results object containing estimates, p-values, etc.
        2. The second return object is the dataframe used for estimation that has problematic columns removed.
        3. A column containing diagnostic information from the different checks and corrections undertaken.
    '''
    # Check for zero trade fixed effects
    data_frame_copy = data_frame.copy()
    adjusted_data_frame, problem_variable_list = _trade_contingent_collinearity_check(data_frame=data_frame_copy,
                                                                                      specification=specification)
    # Check for perfect collinearity
    rhs = adjusted_data_frame.drop(specification.lhs_var, axis=1)
    non_collinear_rhs, collinear_column_list = _collinearity_check(rhs)
    if len(collinear_column_list) == 0:
        collinearity_indicator = 'No'
    else:
        collinearity_indicator = 'Yes'
    collinearity_column = pd.Series({'Collinearities': collinearity_indicator})
    excluded_column_list = problem_variable_list + collinear_column_list
    exclusion_column = pd.Series({'Number of Columns Excluded': len(excluded_column_list)})
    print('Omitted Columns: ' + str(excluded_column_list))
    # GLM Estimation
    try:
        spglm = _SparseGLM(endog=adjusted_data_frame[specification.lhs_var],
                           exog=non_collinear_rhs, family=sm.families.Poisson())
        # start_params: the ordinary least squares estimate by applying the link
        # to valid endogenous values and then solve the normal equations
        valid = (spglm.endog > spglm.family.valid[0]) * (spglm.endog < spglm.family.valid[1])
        olsendog = spglm.family.link(spglm.endog[valid])
        olsexog = spglm.spexog[valid,:]
        olsrhs = olsexog.T.dot(olsendog)
        olsmat = olsexog.T.dot(olsexog)
        start_params = scipy.sparse.linalg.spsolve(olsmat,olsrhs)
        estimates = spglm.fit(cov_type=specification.std_errors,
                              start_params=start_params,
                              maxiter=specification.iteration_limit,
                              method='newton_line_search', max_start_irls=0,
                              extra_fit_funcs={'newton_line_search':_fit_newton_line_search})
        adjusted_data_frame.loc[:,'predicted_trade'] = estimates.mu

        # Checks for overfit (only valid when keep=False)
        fit_check_outcome = _overfit_check(data_frame=adjusted_data_frame,
                                           specification=specification,
                                           predicted_trade_column='predicted_trade')
        overfit_column = pd.Series({'Overfit Warning': fit_check_outcome})

    except:
        traceback.print_exc()
        estimates = 'Estimation could not complete.  GLM process raised an error.'

        # CHECK THAT ESTIMATES DOES NOT GET USED LATER
        # Add something to return for diagnostics. I believe it needs to be a dataframe that can be combined
        overfit_column = pd.Series({'Overfit Warning': 'Estimation could not complete'})


    # Collect diagnostics
    diagnostics = overfit_column.append(collinearity_column)
    diagnostics = diagnostics.append(exclusion_column)
    diagnostics.at['Perfectly Collinear Variables'] = collinear_column_list
    diagnostics.at['Zero Trade Variables'] =  problem_variable_list
    return estimates, adjusted_data_frame, diagnostics


def _trade_contingent_collinearity_check(data_frame, specification):
    '''
    PPML diagnostic for columns that are collinear when trade is greater than zero, as in Santos and Silva (2011)
    Arguments
    :param data_frame: (Pandas.DataFrame) A DataFrame for estimation
    :param specification: (obj) a Specification object from gme.EstimationModel
    :return: (Pandas.DataFrame, list)
        1. A copy of the input data_frame with columns collinear when trade is greater than zero and associated
        observations removed
        2. List containing the names of the columns that were collinear when trade is greater than zero.
    '''

    # Main dataframe for manipulation
    data_frame_copy = data_frame.copy()

    # Identify problematic variables due to perfect collinearity when y>0
    nonzero_data_frame = data_frame_copy.loc[data_frame_copy[specification.lhs_var] > 0,:]
    lhs = [specification.lhs_var]
    rhs_columns = list(nonzero_data_frame.columns)
    rhs_columns.remove(specification.lhs_var)
    rhs = nonzero_data_frame[rhs_columns]
    noncollinear_columns, excluded_columns_list = _collinearity_check(rhs)
    rhs_columns = list(noncollinear_columns.columns)

    # Check if problematic and delete associated observations
    data_frame_copy['mask'] = 1
    problem_variable_list = []
    for col in excluded_columns_list:
        #mean_value = data_frame.loc[(data_frame_copy[specification.lhs_var] > 0),col].mean()
        mean_value = data_frame_copy[data_frame_copy[specification.lhs_var] > 0][col].mean()
        max_value = data_frame_copy[data_frame_copy[specification.lhs_var] == 0][col].max()
        min_value = data_frame_copy[data_frame_copy[specification.lhs_var] == 0][col].min()
        if min_value < mean_value and mean_value < max_value:
            rhs_columns.append(col)
        else:
            problem_variable_list.append(col)
            if data_frame_copy[col].nunique() == 2:
                data_frame_copy.loc[data_frame_copy[col] == 1, 'mask'] = 0

    # Return final data_frame with removed columns and observations
    data_frame_copy = data_frame_copy.loc[data_frame_copy['mask'] != 0,:]
    all = lhs + rhs_columns
    data_frame_copy = data_frame_copy[all]
    return data_frame_copy, problem_variable_list


def _collinearity_check(data_frame, tolerance_level=1e-05):
    '''
    Identifies and drops perfectly collinear columns
    :param data_frame: (Pandas.DataFrame) A DataFrame for estimation
    :param tolerance_level: (float) Tolerance parameter for identifying zero values (default=1e-05)
    :return: (Pandas.DataFrame) Original DataFrame with collinear columns removed
    '''

    data_array = scipy.sparse.csr_matrix(data_frame.values)
    data_array = data_array.T.dot(data_array)
    _, r_factor = scipy.linalg.qr(data_array.toarray(), mode='economic')
    r_diagonal = np.abs(r_factor.diagonal())
    r_range = np.arange(r_diagonal.size)

    # Get list of collinear columns
    collinear_columns = np.where(r_diagonal < tolerance_level)[0]
    collinear_data_frame = data_frame.iloc[:, collinear_columns]
    collinear_column_list = list(collinear_data_frame.columns.values)

    # Get df with independent columns
    collinear_locations = list(collinear_columns)
    independent_columns = list(set(r_range).symmetric_difference(set(collinear_locations)))
    return data_frame.iloc[:, independent_columns], collinear_column_list


# ----------------
# Post-estimation Diagnostics
# ----------------

def _overfit_check(data_frame, specification: str = 'trade',
                   predicted_trade_column: str = 'predicted_trade'):
    '''
    Checks if predictions from GLM estimation are perfectly fitted arguments
    :param data_frame: (Pandas.DataFrame)
    :param specification:
    :param predicted_trade_column:
    :return:
    '''
    '''
    Checks if predictions from GLM estiamtion are perfectly fitted
    Arguments
    ---------
    first: DataFrame object
    second: String for variable name containing trade values (default='trade')
    third: String for variable name containing predicted trade values (defalt='ptrade')

    Returns
    -------
    String indicting if the predictions are perfectly fitted
    '''
    fit_check = 'No'
    non_zero = data_frame.loc[data_frame[specification.lhs_var] > 0,specification.lhs_var]
    low = non_zero.min() * 1e-6
    fit_zero_observations = data_frame.loc[data_frame[specification.lhs_var] == 0, predicted_trade_column]
    if fit_zero_observations.min() < low:
        fit_check = 'Yes'
    return fit_check
