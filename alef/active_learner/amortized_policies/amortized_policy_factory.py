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


from alef.configs.active_learner.amortized_policies.policy_configs import (
    BaseAmortizedPolicyConfig,
    ContinuousGPPolicyConfig,
)
from alef.active_learner.amortized_policies.nn.policies import (
    ContinuousGPPolicy,
)


class AmortizedPolicyFactory:
    @staticmethod
    def build(policy_config: BaseAmortizedPolicyConfig):
        if isinstance(policy_config, ContinuousGPPolicyConfig):
            return ContinuousGPPolicy(**policy_config.dict()).to(policy_config.device)
        else:
            raise NotImplementedError(f"Invalid config: {policy_config.__class__.__name__}")
