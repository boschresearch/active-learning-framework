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

from .modules.essential import LazyDelta

"""
The following code is copied from (with slight modifications):
https://github.com/desi-ivanova/idad/blob/main/neural/aggregators.py
Copyright (c) 2021 Adam Foster, Desi R. Ivanova, Steven Kleinegesse, licensed under the MIT License,
cf. LICENSE file in the root directory of this source tree).
"""


class ImplicitDeepAdaptiveDesign(nn.Module):
    def __init__(self, encoder_network, emission_network, empty_value):
        super().__init__()
        self.encoder = encoder_network
        ## [!] store encoding dim
        self.encoding_dim = encoder_network.output_dim
        self.emitter = emission_network if emission_network is not None else nn.Identity()
        self.register_buffer("prototype", empty_value.clone())
        self.register_parameter("empty_value", nn.Parameter(empty_value))
        # self.empty_value = empty_value

    def lazy(self, *design_obs_pairs):
        module = self  # wrap_parallel(self, self.prototype.is_cuda, dim=0) # split over batch size dim

        def delayed_function():
            return module.forward(*design_obs_pairs)

        lazy_delta = LazyDelta(delayed_function, self.prototype, event_dim=self.prototype.dim())
        return lazy_delta

    def forward(
        self,
    ):
        raise NotImplementedError()


class PermutationInvariantImplicitDAD(ImplicitDeepAdaptiveDesign):
    def __init__(self, encoder_network, emission_network, empty_value, self_attention_layer=None):
        super().__init__(
            encoder_network=encoder_network,
            emission_network=emission_network,
            empty_value=empty_value,
        )
        self.selfattention_layer = self_attention_layer if self_attention_layer is not None else nn.Identity()

    def sum_history_encodings(self, *design_obs_pairs):
        # encode available design-obs pairs, h_t, and stack the representations
        # dimension may be: [t, encoding_dim] or [*batch_size, t, encoding_dim]
        stacked = torch.stack(
            [self.encoder(design, obs) for idx, (design, obs) in enumerate(design_obs_pairs)],
            dim=-2,
        )
        # apply attention (or identity if attention=None)
        batch_size = stacked.shape[:-2]
        stack_size = stacked.shape[-2:]

        stacked = stacked.reshape((-1,) + stack_size)  # force dimension to [batch_size_flatten, t, encoding_dim]
        stacked = self.selfattention_layer(
            stacked
        )  # [batch_size_flatten, t, encoding_dim] -> [batch_size_flatten, t, encoding_dim]
        stacked = stacked.reshape(
            batch_size + stack_size
        )  # turn shape back to [t, encoding_dim] or [*batch_size, t, encoding_dim]
        # sum-pool the resulting encodings across t (dim=-2)
        # dimension is: [encoding_dim] or [*batch_size, encoding_dim]
        sum_encoding = stacked.sum(dim=-2)

        return sum_encoding

    def empty_forward(self):
        # For efficiency: learn the first design separately, i.e. do not pass a
        # vector (e.g. of 0s) through the emitter network.
        # !This doesn't affect critic net, since len(design_obs_pairs) is never 0.
        output = self.empty_value
        #### To pass a zero vector though the emitter, use this: ###
        # zero_vec = self.empty_value.new_zeros(self.encoding_dim)
        # output = self.emitter(zero_vec)
        return output

    def forward(self, *design_obs_pairs):
        if len(design_obs_pairs) == 0:
            output = self.empty_forward()
        else:
            sum_encoding = self.sum_history_encodings(*design_obs_pairs)
            output = self.emitter(sum_encoding)
        return output


class LSTMImplicitDAD(ImplicitDeepAdaptiveDesign):
    def __init__(self, encoder_network, emission_network, empty_value, num_hidden_layers=2):
        super().__init__(encoder_network, emission_network, empty_value)
        self.lstm_net = nn.LSTM(self.encoding_dim, self.encoding_dim, num_hidden_layers, batch_first=True)

    def lstm_history_encodings(self, *design_obs_pairs):
        # Input to LSTM should be [batch, seq, feature]

        # encode available design-obs pairs, h_t, and stack the representations
        # dimension is: [batch_size, t, encoding_dim]
        stacked = torch.stack(
            [self.encoder(design, obs, t=[idx + 1]) for idx, (design, obs) in enumerate(design_obs_pairs)],
            dim=-2,
        )
        # keep the last state
        _, (h_n, c_n) = self.lstm_net(stacked)
        # return the hidden state from the last layer
        # dimension [batch_size, encoding_dim]
        return h_n[-1]

    def empty_forward(self):
        # pass zeros to the LSTM if no history is available yet
        stacked = self.empty_value.new_zeros(1, 1, self.encoding_dim)
        # keep the last state
        _, (h_n, c_n) = self.lstm_net(stacked)
        # return the hidden state from the last layer
        # dimension [batch_size, encoding_dim]
        return self.emitter(h_n[-1])

    def forward(self, *design_obs_pairs):
        if len(design_obs_pairs) == 0:
            output = self.empty_forward()
        else:
            lstm_encoding = self.lstm_history_encodings(*design_obs_pairs)
            output = self.emitter(lstm_encoding)
        return output
