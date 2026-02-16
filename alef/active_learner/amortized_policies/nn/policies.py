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

import torch
from torch import nn
from typing import Sequence, Union, Tuple

from .aggregators import PermutationInvariantImplicitDAD
from .modules.essential import GeneralizedSigmoid, GeneralizedTanh
from .modules.history_encoder import MLPHistoryEncoder
from .modules.selfattention import SelfAttention
from .modules.emitter import MLPEmitter
from alef.configs.base_parameters import INPUT_DOMAIN
from alef.enums.active_learner_amortized_policy_enums import DomainWarpperType


class ContinuousGPPolicy(PermutationInvariantImplicitDAD):
    def __init__(
        self,
        input_dim: int,
        observation_dim: int,
        *,
        hidden_dim_encoder: Union[int, Sequence[int]],
        encoding_dim: int,
        hidden_dim_emitter: Union[int, Sequence[int]],
        input_domain: Tuple[Union[int, float], Union[int, float]] = INPUT_DOMAIN,
        activation: nn.Module = nn.Softplus(),
        self_attention_layer: bool = False,
        domain_warpper: DomainWarpperType = DomainWarpperType.TANH,
        **kwargs,
    ):
        if domain_warpper == DomainWarpperType.SIGMOID:
            damain_warpping_layer = GeneralizedSigmoid
        elif domain_warpper == DomainWarpperType.TANH:
            damain_warpping_layer = GeneralizedTanh
        else:
            raise ValueError
        history_encoder = MLPHistoryEncoder(
            input_dim=(1, input_dim),
            observation_dim=observation_dim,
            hidden_dim=hidden_dim_encoder,
            output_dim=encoding_dim,
            activation=activation,
            output_activation=nn.Identity(),
            name="policy_history_encoder",
        )
        design_emitter = MLPEmitter(
            input_dim=encoding_dim,
            hidden_dim=hidden_dim_emitter,
            output_dim=(1, input_dim),
            activation=activation,
            output_activation=damain_warpping_layer(torch.tensor(input_domain, dtype=torch.float)),
            name="policy_design_emitter",
        )
        empty_value = torch.zeros((1, input_dim))
        # Design net: takes pairs [design, observation] as input
        super().__init__(
            history_encoder,
            design_emitter,
            empty_value=empty_value,
            self_attention_layer=SelfAttention(encoding_dim, encoding_dim) if self_attention_layer else None,
        )
        self.register_buffer("input_domain", torch.tensor(input_domain, dtype=torch.float))

    def empty_forward(self):
        a, b = self.input_domain
        return torch.rand_like(self.empty_value) * (b - a) + a
