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
        print("fixed effects done")
        estimating_data_frame = pd.concat(
            [data_frame[specification.lhs_var], data_frame[specification.rhs_var], fixed_effects_df], axis=1)
        print("concat done")
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

from petsc4py import PETSc

class WLSMat(object):
    def __init__(self, A, w):
        self._wlsA = A
        self._wlsw = w

    def mult(self, mat, x, y):
        z = self._wlsw.duplicate()
        self._wlsA.mult(x, z)
        z.pointwiseMult(z,self._wlsw)
        self._wlsA.multTranspose(z, y)


class PPML(object):
    """
    A^T (exp(A x) - y) = 0
    """

    def __init__(self, csr_mat, rhs):
        m, n = csr_mat.shape
        nz_per_row = csr_mat.indptr[1:] - csr_mat.indptr[:-1]
        indptr = csr_mat.indptr.copy()
        indices = csr_mat.indices.copy()
        values = csr_mat.data.copy()
        self.rhs = PETSc.Vec().createWithArray(rhs.copy(), size=m)
        self.rhs.view()
        self.mat = PETSc.Mat().createAIJWithArrays([m,n], (indptr, indices, values))
        self.mat.view()

    def formFunction(self, snes, X, F):
        Y = self.rhs.duplicate()
        self.mat.mult(X, Y)
        Y.exp()
        Y.axpy(-1.,self.rhs)
        self.mat.multTranspose(Y, F)

    def formJacobian(self, snes, X, J, P):
        wls = J.getPythonContext()
        wls._wlsA = self.mat
        self.mat.mult(X, wls._wlsw)
        wls._wlsw.exp()


class _GLM(sm.GLM):
    """
    The same as a statsmodels Generalized Linear Model (GLM), but
    using sparse linear algebra from scipy within the iteratively reweighted
    least squares (IRLS) algorithm
    """
    def _fit_irls(self, start_params=None, maxiter=100, tol=1e-8,
                  scale=None, cov_type='nonrobust', cov_kwds=None,
                  use_t=None, **kwargs):
        """
        Fits a generalized linear model for a given family using
        iteratively reweighted least squares (IRLS).
        """

        import petsc4py
        import statsmodels.regression.linear_model as lm
        from statsmodels.genmod.generalized_linear_model import _check_convergence


        attach_wls = kwargs.pop('attach_wls', False)
        atol = kwargs.get('atol')
        rtol = kwargs.get('rtol', 0.)
        tol_criterion = kwargs.get('tol_criterion', 'deviance')
        wls_method = kwargs.get('wls_method', 'lstsq')
        atol = tol if atol is None else atol

        endog = self.endog
        wlsexog = scipy.sparse.csr_matrix(self.exog)
        ppml = PPML(wlsexog, endog)
        snes = PETSc.SNES().create()
        wlsmat = WLSMat(ppml.mat, ppml.rhs.duplicate())
        J = PETSc.Mat().createPython([wlsexog.shape[1], wlsexog.shape[1]], context=wlsmat)
        snes.setFunction(ppml.formFunction, ppml.mat.createVecRight())
        snes.setJacobian(ppml.formJacobian, J)
        snes.setFromOptions()
        sol = ppml.mat.createVecRight()
        snes.solve(None, sol)
        sol.view()
        
        if start_params is None:
            start_params = np.ones(self.exog.shape[1])
        sol = start_params
        lin_pred = wlsexog.dot(start_params) + self._offset_exposure
        mu = self.family.fitted(lin_pred)
        self.scale = self.estimate_scale(mu)
        dev = self.family.deviance(self.endog, mu, self.var_weights,
                                   self.freq_weights, self.scale)
        if np.isnan(dev):
            raise ValueError("The first guess on the deviance function "
                             "returned a nan.  This could be a boundary "
                             " problem and should be reported.")

        # first guess on the deviance is assumed to be scaled by 1.
        # params are none to start, so they line up with the deviance
        history = dict(params=[np.inf, start_params], deviance=[np.inf, dev])
        converged = False
        criterion = history[tol_criterion]
        # This special case is used to get the likelihood for a specific
        # params vector.
        if maxiter == 0:
            mu = self.family.fitted(lin_pred)
            self.scale = self.estimate_scale(mu)
            wls_results = lm.RegressionResults(self, start_params, None)
            sol = wls_results.params
            iteration = 0
        for iteration in range(maxiter):
            print(iteration)
            self.weights = (self.iweights * self.n_trials *
                            self.family.weights(mu))
            wlsendog = (self.family.link.deriv(mu) * (self.endog-mu)
                        - self._offset_exposure)
            wls_mod = _MinimalWLS(wlsendog, wlsexog,
                                  self.weights, check_endog=True,
                                  check_weights=True)
            wls_results = wls_mod.fit(method=wls_method)
            sol += wls_results.params
            lin_pred = np.dot(self.exog, sol)
            lin_pred += self._offset_exposure
            mu = self.family.fitted(lin_pred)
            print(np.linalg.norm(self.exog.T.dot(self.endog - mu)))
            history = self._update_history(wls_results, mu, history)
            self.scale = self.estimate_scale(mu)
            if endog.squeeze().ndim == 1 and np.allclose(mu - endog, 0):
                msg = "Perfect separation detected, results not available"
                raise PerfectSeparationError(msg)
            converged = _check_convergence(criterion, iteration + 1, atol,
                                           rtol)
            if converged:
                break
        self.mu = mu

        if maxiter > 0:  # Only if iterative used
            wls_method2 = 'pinv' if wls_method == 'lstsq' else wls_method
            wls_model = lm.WLS(wlsendog, wlsexog, self.weights)
            wls_results = wls_model.fit(method=wls_method2)

        glm_results = sm.GLMResults(self, wls_results.params,
                                    wls_results.normalized_cov_params,
                                    self.scale,
                                    cov_type=cov_type, cov_kwds=cov_kwds,
                                    use_t=use_t)

        glm_results.method = "IRLS"
        glm_results.mle_settings = {}
        glm_results.mle_settings['wls_method'] = wls_method
        glm_results.mle_settings['optimizer'] = glm_results.method
        if (maxiter > 0) and (attach_wls is True):
            glm_results.results_wls = wls_results
        history['iteration'] = iteration + 1
        glm_results.fit_history = history
        glm_results.converged = converged
        return sm.GLMResultsWrapper(glm_results)


class _MinimalWLS(sm.regression._tools._MinimalWLS):
    """
    The same as a statsmodels.regression._tools._MinimalWLS, but
    using sparse linear algebra from scipy for the least squares problem
    """
    
    def __init__(self, endog, exog, weights=1.0, check_endog=False,
                 check_weights=False):
        self.endog = endog
        self.exog = exog
        self.weights = weights
        w_half = np.sqrt(weights)
        if check_weights:
            if not np.all(np.isfinite(w_half)):
                raise ValueError(self.msg.format('weights'))

        if check_endog:
            if not np.all(np.isfinite(endog)):
                raise ValueError(self.msg.format('endog'))

        self.wendog = w_half * endog
        if np.isscalar(weights):
            self.wexog = w_half * exog
        else:
            self.wexog = scipy.sparse.spdiags(w_half, 0, len(w_half), len(w_half)) @ exog

    def fit(self, method='pinv'):
        print(np.linalg.norm(self.wendog),np.linalg.norm(self.wexog.T.dot(self.wendog)))
        result = scipy.sparse.linalg.lsmr(self.wexog, self.wendog, atol=1.e-16, btol=1.e-16)
        #result = np.linalg.lstsq(self.wexog.toarray(), self.wendog, rcond=-1)
        return self.results(result[0])



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

    print("trade contingent collinearity check done")
    # Check for perfect collinearity
    rhs = adjusted_data_frame.drop(specification.lhs_var, axis=1)
    non_collinear_rhs, collinear_column_list = _collinearity_check(rhs)
    print("collinearity check done")
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
        glm = _GLM(endog=adjusted_data_frame[specification.lhs_var],
                   exog=non_collinear_rhs,
                   family=sm.families.Poisson())
        estimates = glm.fit(cov_type=specification.std_errors, maxiter=specification.iteration_limit)
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

    data_array = data_frame.values
    q_factor, r_factor = np.linalg.qr(data_array)
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
