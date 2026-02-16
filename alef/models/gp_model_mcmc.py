# Copyright (c) 2024 Robert Bosch GmbH
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import logging
from alef.models.gp_model import GPModel
from alef.models.base_model import BaseModel
import numpy as np
from typing import Tuple, Optional
from alef.models.gp_model_mcmc_proposals import BaseGpModelMCMCProposal
from alef.models.gp_model_laplace import GPModelLaplace, PredictionType
from enum import Enum
from alef.utils.gaussian_mixture_density import GaussianMixtureDensity

logger = logging.getLogger(__name__)


class InternalInferenceType(Enum):
    MAP = 1
    LAPLACE = 2


class GPModelMCMC(BaseModel):
    """
    Based on "Discovering and Exploiting Additive Structure for Bayesian Optimization" (Gardner et al. 2017). This class produces as model a Bayesian Model Average
    over multiple Gaussian Processes by performing MCMC sampling over kernels. It uses Metropolis Hastings as MCMC method that samples models/kernel from the posterior.
    It therefore needs a proposal over models/kernels which it gets as BaseGpModelMCMCProposal object - these objects also define the prior over models.
    Calling the infer method will start the sampling over the model. The prediction is than done via the standard prediction for BMA's.

    Attributes:
        proposal - BaseGpModelMCMCProposal object acting as proposal distribution over kernels and also specifiying the prior over kernels
        initial_state - BaseGpModelMCMCState object specifying the initial state of the chain - set to the current state of the proposal at initialization
        n_samples - int number of posterios samples
        n_burnin - int number of burnins that are thrown away
        n_thinned - int specifing how many (-1) samples are discarded in the regular chain (after burn in)
        internal_inference_type - InternalInferenceType - MH needs the marginal likelihood of the model/kernel M: p(y|x,M). This can be approximated via MAP estimation or via Laplace and set with this variable.
    """

    def __init__(
        self,
        proposal: BaseGpModelMCMCProposal,
        internal_inference_type: InternalInferenceType,
        n_samples: int,
        n_burnin: int,
        n_thinned: int,
        train_likelihood_variance: bool,
        initial_observation_noise: float,
        expected_observation_noise: float,
        perform_multi_start_opt_in_each_step: bool,
        **kwargs,
    ) -> None:
        self.proposal = proposal
        self.initial_state = self.proposal.get_current_state()
        self.n_samples = n_samples
        self.n_burnin = n_burnin
        self.n_thinned = n_thinned
        self.n_complete = self.n_burnin + self.n_thinned * self.n_samples
        self.posterior_samples = []
        self.initial_observation_noise = initial_observation_noise
        self.expected_observation_noise = expected_observation_noise
        self.train_likelihood_variance = train_likelihood_variance
        self.internal_inference_type = internal_inference_type
        self.perform_multi_start_opt_in_each_step = perform_multi_start_opt_in_each_step

    def reset_model(self):
        """
        resets the model to internal states that were present at initialization - initial state of the MC chain
        """
        self.proposal.set_current_state(self.initial_state)

    def infer(self, x_data: np.array, y_data: np.array):
        """
        Performs inference of the model - Creates samples from the model posterior p(M|y_data,x_data) propto p(y_data|x_data,M)P(M) using MCMC/MH

        Arguments:
        x_data: Input array with shape (n,d) where d is the input dimension and n the number of training points
        y_data: Label array with shape (n,1) where n is the number of training points
        """

        for i in range(0, self.n_complete):
            logger.info("Step " + str(i) + " of " + str(self.n_complete) + " with " + str(self.n_burnin) + " burn ins")
            current_state = self.proposal.get_current_state()
            proposed_state, p_m_prop_m_curr, p_m_curr_m_prop = self.proposal.propose_next()
            if self.internal_inference_type == InternalInferenceType.MAP:
                current_model = GPModel(
                    kernel=current_state.get_kernel(),
                    optimize_hps=True,
                    perform_multi_start_optimization=self.perform_multi_start_opt_in_each_step,
                    observation_noise=self.initial_observation_noise,
                    expected_observation_noise=self.expected_observation_noise,
                    train_likelihood_variance=self.train_likelihood_variance,
                    set_prior_on_observation_noise=True,
                )
                proposed_model = GPModel(
                    kernel=proposed_state.get_kernel(),
                    optimize_hps=True,
                    perform_multi_start_optimization=self.perform_multi_start_opt_in_each_step,
                    observation_noise=self.initial_observation_noise,
                    expected_observation_noise=self.expected_observation_noise,
                    train_likelihood_variance=self.train_likelihood_variance,
                    set_prior_on_observation_noise=True,
                )
                current_model.deactivate_summary_printing()
                proposed_model.deactivate_summary_printing()
            elif self.internal_inference_type == InternalInferenceType.LAPLACE:
                current_model = GPModelLaplace(
                    kernel=current_state.get_kernel(),
                    observation_noise=self.initial_observation_noise,
                    perform_multi_start_optimization=self.perform_multi_start_opt_in_each_step,
                    expected_observation_noise=self.expected_observation_noise,
                    train_likelihood_variance=self.train_likelihood_variance,
                    prediction_type=PredictionType.MAP,
                )
                proposed_model = GPModelLaplace(
                    kernel=proposed_state.get_kernel(),
                    observation_noise=self.initial_observation_noise,
                    perform_multi_start_optimization=self.perform_multi_start_opt_in_each_step,
                    expected_observation_noise=self.expected_observation_noise,
                    train_likelihood_variance=self.train_likelihood_variance,
                    prediction_type=PredictionType.MAP,
                )
            current_model.infer(x_data, y_data)
            proposed_model.infer(x_data, y_data)
            current_model_log_evidence = current_model.model.log_posterior_density()
            proposed_model_log_evidence = proposed_model.model.log_posterior_density()
            logger.debug("Current model log evidence")
            logger.debug(current_model_log_evidence)
            logger.debug("Proposed model log evidence")
            logger.debug(proposed_model_log_evidence)
            is_uniform, current_model_prior_probability = current_state.get_prior_probability()
            _, proposed_model_prior_probability = proposed_state.get_prior_probability()
            if is_uniform:
                current_model_prior_probability = 1.0
                proposed_model_prior_probability = 1.0
            logger.debug("Current model prior probability:")
            logger.debug(current_model_prior_probability)
            logger.debug("Proposed model prior probability:")
            logger.debug(proposed_model_prior_probability)
            sample = np.random.random(1)
            accepted = self.check_acceptance(
                sample,
                proposed_model_log_evidence,
                current_model_log_evidence,
                p_m_curr_m_prop,
                p_m_prop_m_curr,
                proposed_model_prior_probability,
                current_model_prior_probability,
            )
            if accepted:
                logger.info("-ACCEPT")
                self.proposal.accept()
                if i >= self.n_burnin and (i - self.n_burnin) % self.n_thinned == 0:
                    self.posterior_samples.append((proposed_state, proposed_model))
            else:
                logger.info("-REJECT")
                if i >= self.n_burnin and (i - self.n_burnin) % self.n_thinned == 0:
                    self.posterior_samples.append((current_state, current_model))

    def check_acceptance(
        self,
        sample: float,
        proposed_model_log_evidence: float,
        current_model_log_evidence: float,
        p_m_curr_m_prop: float,
        p_m_prop_m_curr: float,
        proposed_model_prior_probability: float,
        current_model_prior_probability: float,
    ):
        """
        Helper method that implements the Metropolis hasting acceptance decision
        """
        log_acceptance_probability = min(
            0.0,
            proposed_model_log_evidence
            + np.log(proposed_model_prior_probability)
            + np.log(p_m_curr_m_prop)
            - current_model_log_evidence
            - np.log(current_model_prior_probability)
            - np.log(p_m_prop_m_curr),
        )
        return np.log(sample) <= log_acceptance_probability

    def predict(self, x_test: np.array):
        """
        creates lists of predictive mu and sigma arrays associated with the posterior model/kernel samples
        """
        pred_mus = []
        pred_sigmas = []
        for _, model in self.posterior_samples:
            pred_mu_model, pred_sigma_model = model.predictive_dist(x_test)
            pred_mus.append(pred_mu_model)
            pred_sigmas.append(pred_sigma_model)
        pred_mus_complete = np.array(pred_mus)
        pred_sigmas_complete = np.array(pred_sigmas)
        return pred_mus_complete, pred_sigmas_complete

    def predictive_dist(self, x_test: np.array) -> Tuple[np.array, np.array]:
        """
        Method for retrieving the predictive mean and sigma for a given array of the test points
        Computes predictive distribution of the Bayesian Model Average --> creates a Gaussian Mixture distribution
        and retrieves the mean and sigma of that distribution

        Arguments:
        x_test: Array of test input points with shape (n,d) where d is the input dimension and n the number of test points

        Returns:
        mean array with shape either (n,)
        sigma array with shape either (n,)
        """
        logger.info("-PREDICT")
        pred_mus_complete, pred_sigmas_complete = self.predict(x_test)
        n = x_test.shape[0]
        mus_over_inputs = []
        sigmas_over_inputs = []
        for i in range(0, n):
            mu = np.mean(pred_mus_complete[:, i])
            var = np.mean(
                np.power(pred_mus_complete[:, i], 2.0) + np.power(pred_sigmas_complete[:, i], 2.0) - np.power(mu, 2.0)
            )
            mus_over_inputs.append(mu)
            sigmas_over_inputs.append(np.sqrt(var))
        return np.array(mus_over_inputs), np.array(sigmas_over_inputs)

    def entropy_predictive_dist(self, x_test: np.array) -> np.array:
        """
        Method for calculating the entropy of the predictive distribution for test sequence - used for acquistion function in active learning
        Calculates the entropy of the predictive distribution associdated with the BMA over sampled models --> entropy of a Gaussian Mixture

        Arguments:
        x_test: Array of test input points with shape (n,d) where d is the input dimension and n the number of test points

        Returns:
        entropy array with shape (n,1)
        """

        pred_mus_complete, pred_sigmas_complete = self.predict(x_test)
        n = x_test.shape[0]
        m_posterior_draws = len(pred_mus_complete[:, 0])
        assert m_posterior_draws == len(self.posterior_samples)
        weights = np.repeat(1 / m_posterior_draws, m_posterior_draws)
        entropies = []
        for i in range(0, n):
            gmm_at_test_point = GaussianMixtureDensity(weights, pred_mus_complete[:, i], pred_sigmas_complete[:, i])
            entropy = gmm_at_test_point.entropy()
            logger.debug(str(i) + "/" + str(n) + ":")
            logger.debug(entropy)
            entropies.append(entropy)
        return np.array(entropies)

    def estimate_model_evidence(self, x_data: Optional[np.array], y_data: Optional[np.array]) -> np.float:
        raise NotImplementedError

    def predictive_log_likelihood(self, x_test: np.array, y_test: np.array) -> np.array:
        """
        Method for calculating the log likelihood value of the the predictive distribution at the test input points (evaluated at the output values)
        - method is therefore for validation purposes only

        Arguments:
        x_test: Array of test input points with shape (n,d) where d is the input dimension and n the number of test points
        y_test: Array of test output points with shape (n,1)

        Returns:
        array of shape (n,) with log liklihood values
        """

        pred_mus_complete, pred_sigmas_complete = self.predict(x_test)
        n = x_test.shape[0]
        m_posterior_draws = len(pred_mus_complete[:, 0])
        assert m_posterior_draws == len(self.posterior_samples)
        weights = np.repeat(1 / m_posterior_draws, m_posterior_draws)
        log_likelis = []
        for i in range(0, n):
            gmm_at_test_point = GaussianMixtureDensity(weights, pred_mus_complete[:, i], pred_sigmas_complete[:, i])
            log_likeli = gmm_at_test_point.log_likelihood(np.squeeze(y_test[i]))
            if np.isinf(log_likeli):
                logger.debug(y_test[i])
                gmm_at_test_point.plot(y=y_test[i])
            log_likelis.append(log_likeli)
        return np.squeeze(np.array(log_likelis))
