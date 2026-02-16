from alef.configs.models.gp_model_mcmc_config import (
    AdditiveProposalGPModelMCMCConfig,
    KernelGrammarGPModelMCMCConfig,
)
from alef.models.model_factory import ModelFactory
from alef.configs.kernels.rbf_configs import RBFWithPriorConfig
import pytest
import gpflow
import numpy as np


def test_model_mcmc_additive():
    model_config = AdditiveProposalGPModelMCMCConfig(input_dimension=4, initial_partition_list=[[2, 1], [0], [3]])
    model = ModelFactory.build(model_config)
    proposal = model.proposal
    proposed_state, _, _ = proposal.propose_next()
    partition = proposed_state.partition
    dims_in_partition = np.sum([len(element) for element in partition.partition_list])
    assert dims_in_partition == 4
    proposal.accept()
    assert proposed_state == proposal.get_current_state()
    proposed_state, _, _ = proposal.propose_next()
    assert not proposed_state == proposal.get_current_state()
    assert isinstance(proposed_state.get_kernel(), gpflow.kernels.Kernel)


def test_model_mcmc_kernel_grammar():
    kernel_config = RBFWithPriorConfig(input_dimension=4)
    model_config = KernelGrammarGPModelMCMCConfig(
        input_dimension=4, initial_base_kernel_config=kernel_config, add_hp_prior=True
    )
    model = ModelFactory.build(model_config)
    proposal = model.proposal
    proposed_state, _, _ = proposal.propose_next()
    proposal.accept()
    assert proposed_state == proposal.get_current_state()
    proposed_state, _, _ = proposal.propose_next()
    assert not proposed_state == proposal.get_current_state()
    _, prob = proposed_state.get_prior_probability()
    assert prob >= 0.0 and prob <= 1.0
    assert isinstance(proposed_state.get_kernel(), gpflow.kernels.Kernel)


@pytest.mark.parametrize("factor1,factor2", [(0.1, 0.8), (0.2, 0.5), (0.5, 0.5)])
def test_mh_in_gp_mcmc(factor1, factor2):
    kernel_config = RBFWithPriorConfig(input_dimension=4)
    model_config = KernelGrammarGPModelMCMCConfig(
        input_dimension=4, initial_base_kernel_config=kernel_config, add_hp_prior=True
    )
    model = ModelFactory.build(model_config)
    prior_proposed = factor1
    prior_current = factor2
    log_evidence_proposed = np.log(factor2)
    log_evidence_current = np.log(factor1)
    p_m_curr_m_prob = 0.4
    p_m_prob_m_curr = 0.5
    sample = 0.7
    assert model.check_acceptance(
        sample,
        log_evidence_proposed,
        log_evidence_current,
        p_m_curr_m_prob,
        p_m_prob_m_curr,
        prior_proposed,
        prior_current,
    )
    sample = 0.9
    assert not model.check_acceptance(
        sample,
        log_evidence_proposed,
        log_evidence_current,
        p_m_curr_m_prob,
        p_m_prob_m_curr,
        prior_proposed,
        prior_current,
    )


if __name__ == "__main__":
    test_mh_in_gp_mcmc()
