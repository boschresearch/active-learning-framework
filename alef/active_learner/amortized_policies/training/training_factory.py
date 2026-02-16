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

from typing import Union
import pyro

from alef.configs.active_learner.amortized_policies.loss_configs import (
    BasicAmortizedPolicyLossConfig,
    BasicLossCurriculumConfig,
    ContinuousDADLossConfig,
    ContinuousScoreDADLossConfig,
    GPEntropyLoss1Config,
    GPEntropyLoss2Config,
    GPMILoss1Config,
    GPMILoss2Config,
)
from alef.active_learner.amortized_policies.loss.curriculum import (
    TrivialLossCurriculum,
    LossCurriculum,
)
from alef.active_learner.amortized_policies.loss.multiple_steps.oed_mi import (
    PriorContrastiveEstimation,
    PriorContrastiveEstimationScoreGradient,
)
from alef.active_learner.amortized_policies.simulated_processes.multiple_steps.continuous_gp_al import (
    SequentialGaussianProcessContinuousDomain,
)
from alef.active_learner.amortized_policies.amortized_policy_factory import AmortizedPolicyFactory
from alef.active_learner.amortized_policies.loss.multiple_steps.gp_entropy import (
    GPEntroopy1,
    GPEntroopy2,
)
from alef.active_learner.amortized_policies.loss.multiple_steps.gp_mi import (
    GPMutualInformation1,
    GPMutualInformation2,
)
from alef.active_learner.amortized_policies.training.dad_oed import OED

# from alef.active_learner.amortized_policies.training.idad_oed import OED
from alef.configs.active_learner.amortized_policies.training_configs import (
    BaseAmortizedPolicyTrainingConfig,
    AmortizedNonmyopicContinuousFixGPPolicyTrainingConfig,
    AmortizedNonmyopicContinuousRandomGPPolicyTrainingConfig,
)


class _LossClassPicker:
    @staticmethod
    def pick_loss_class(loss_config: BasicAmortizedPolicyLossConfig):
        if isinstance(loss_config, ContinuousDADLossConfig):
            return PriorContrastiveEstimation
        elif isinstance(loss_config, ContinuousScoreDADLossConfig):
            return PriorContrastiveEstimationScoreGradient
        elif isinstance(loss_config, GPEntropyLoss1Config):
            return GPEntroopy1
        elif isinstance(loss_config, GPEntropyLoss2Config):
            return GPEntroopy2
        elif isinstance(loss_config, GPMILoss1Config):
            return GPMutualInformation1
        elif isinstance(loss_config, GPMILoss2Config):
            return GPMutualInformation2
        else:
            raise NotImplementedError(f"Invalid config: {loss_config.__class__.__name__}")


class _LossClassSetter:
    @staticmethod
    def get_loss_class(loss_config: Union[BasicAmortizedPolicyLossConfig, BasicLossCurriculumConfig]):
        if isinstance(loss_config, BasicAmortizedPolicyLossConfig):
            return TrivialLossCurriculum(
                _LossClassPicker.pick_loss_class(loss_config)(
                    **loss_config.dict(exclude={"num_epochs", "epochs_size"})
                ),
                **loss_config.dict(include={"num_epochs", "epochs_size"}),
            )
        elif isinstance(loss_config, BasicLossCurriculumConfig):
            loss_list = []
            num_epochs_list = []
            epochs_size_list = []
            for individual_loss_config in loss_config.loss_config_list:
                loss_list.append(
                    _LossClassPicker.pick_loss_class(individual_loss_config)(
                        **individual_loss_config.dict(exclude={"num_epochs", "epochs_size"})
                    )
                )
                num_epochs_list.append(individual_loss_config.num_epochs)
                epochs_size_list.append(individual_loss_config.epochs_size)

            return LossCurriculum(loss_list, num_epochs_list, epochs_size_list)
        else:
            raise NotImplementedError(f"Invalid config: {loss_config.__class__.__name__}")


class AmortizedLearnerTrainingFactory:
    @staticmethod
    def build(training_config: BaseAmortizedPolicyTrainingConfig):
        if isinstance(
            training_config,
            (
                AmortizedNonmyopicContinuousFixGPPolicyTrainingConfig,
                AmortizedNonmyopicContinuousRandomGPPolicyTrainingConfig,
            ),
        ):
            # Annealed LR optimiser --------
            optimizer = training_config.optimizer
            scheduler = pyro.optim.ExponentialLR(
                {
                    "optimizer": optimizer,
                    "optim_args": training_config.optim_args,
                    "gamma": training_config.gamma,
                }
            )
            process = SequentialGaussianProcessContinuousDomain(
                AmortizedPolicyFactory.build(training_config.policy_config),
                kernel_config=training_config.kernel_config,
                n_steps=training_config.n_steps,
                sample_gp_prior=training_config.sample_gp_prior,
                device=training_config.policy_config.device,
            )
            pce_loss = _LossClassSetter.get_loss_class(training_config.loss_config)

            return OED(process, scheduler, pce_loss)
        else:
            raise NotImplementedError(f"Invalid config: {training_config.loss_config.__class__.__name__}")
