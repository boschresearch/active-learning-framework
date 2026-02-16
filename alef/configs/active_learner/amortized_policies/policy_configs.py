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

from typing import Tuple, Sequence, Union
from pydantic import BaseSettings


from alef.configs.base_parameters import INPUT_DOMAIN
from alef.enums.active_learner_amortized_policy_enums import DomainWarpperType

__all__ = [
    "ContinuousGPPolicyConfig",
]


class BaseAmortizedPolicyConfig(BaseSettings):
    input_domain: Tuple[Union[int, float], Union[int, float]] = INPUT_DOMAIN
    name: str = "basic_al_policy_config"


class ContinuousGPPolicyConfig(BaseAmortizedPolicyConfig):
    input_dim: int
    observation_dim: int = 1
    hidden_dim_encoder: Union[int, Sequence[int]] = 512
    encoding_dim: int = 32
    hidden_dim_emitter: Union[int, Sequence[int]] = 512
    input_domain: Tuple[Union[int, float], Union[int, float]] = INPUT_DOMAIN
    # activation: nn.Module=nn.Softplus()
    self_attention_layer: bool = True
    domain_warpper: DomainWarpperType = DomainWarpperType.TANH
    device: str = "cpu"
    name: str = "basic_continuous_gp_al_policy_config"
