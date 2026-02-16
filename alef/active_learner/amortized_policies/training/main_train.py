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

import datetime
import numpy as np
import pandas as pd
from pathlib import Path

import torch
from pyro.infer.util import torch_item
import pyro
from tqdm import trange

from alef.active_learner.amortized_policies.training.training_factory import AmortizedLearnerTrainingFactory
from alef.active_learner.amortized_policies.loss.curriculum import BaseCurriculum
from alef.configs.active_learner.amortized_policies import loss_configs
from alef.configs.active_learner.amortized_policies import training_configs
from alef.configs.active_learner.amortized_policies.policy_configs import (
    ContinuousGPPolicyConfig,
)

# need GP model
from alef.enums.active_learner_amortized_policy_enums import DomainWarpperType
from alef.utils.utils import write_dict_to_json


class Trainer:
    def __init__(
        self,
        log_path,
        seed: int,
        device: str,
        fast_tqdm: bool,
    ):
        self.root_path = Path(log_path)
        self.root_path.mkdir(exist_ok=True)
        self.root_path = self.root_path / (
            "%s_seed%s" % (datetime.datetime.now().astimezone().strftime("%Z_%Y_%m_%d__%H_%M_%S"), seed)
        )
        self.root_path.mkdir(exist_ok=True)

        self.params_path = self.root_path / "params"
        self.params_path.mkdir(exist_ok=True)
        self.result_path = self.root_path / "result"
        self.result_path.mkdir(exist_ok=True)

        self.seed = seed
        self.device = device
        self.fast_tqdm = fast_tqdm

        if device.startswith("cuda"):
            n_devices = torch.cuda.device_count()
            for d in range(n_devices):
                # warm up each GPU
                _ = torch.tensor([], device=d)
                _ = torch.linalg.cholesky(torch.ones((0, 0), device=d))
                print(f"warm up GPU {d}")

    def set_policy_config(self, input_dimension: int, self_attention_layer: bool, domain_warpper: DomainWarpperType):
        self.policy_config = ContinuousGPPolicyConfig(
            input_dim=input_dimension,
            observation_dim=1,
            self_attention_layer=self_attention_layer,
            domain_warpper=domain_warpper,
            device=self.device,
        )

    def set_loss_config(self, loss_config_name: str):
        self.loss_config = getattr(loss_configs, loss_config_name)()

    def set_training_config(self, training_config_name: str, kernel_config, T: int):
        # number of queries during training
        self.kernel_config = kernel_config

        self.training_config = getattr(training_configs, training_config_name)(
            policy_config=self.policy_config,
            kernel_config=self.kernel_config,
            n_steps=T,
            loss_config=self.loss_config,
        )

    def save_settings(self):
        write_dict_to_json(
            self.policy_config.json(), self.params_path / f"policy_config_{self.policy_config.__class__.__name__}.json"
        )
        write_dict_to_json(
            self.loss_config.json(), self.params_path / f"loss_config_{self.loss_config.__class__.__name__}.json"
        )
        write_dict_to_json(
            self.kernel_config.json(), self.params_path / f"kernel_config_{self.kernel_config.__class__.__name__}.json"
        )
        write_dict_to_json(
            self.training_config.json(exclude={"policy_config", "kernel_config", "loss_config"}),
            self.params_path / f"training_config_{self.training_config.__class__.__name__}.json",
        )

    def train(self):
        pyro.clear_param_store()
        pyro.set_rng_seed(self.seed)
        np.random.seed(self.seed)

        assert hasattr(self, "policy_config")
        assert hasattr(self, "loss_config")
        assert hasattr(self, "training_config")

        self.save_settings()
        self.oed = AmortizedLearnerTrainingFactory.build(self.training_config)
        assert isinstance(self.oed.loss, BaseCurriculum)

        self.loss_history = []
        self.check_point_overview = pd.DataFrame(columns=["loss", "epoch_mean_loss", "rmse_mean", "rmse_stderr"])

        loss_str = "Loss:   0.000 "
        if self.fast_tqdm:
            t = trange(1, self.oed.loss.num_steps + 1, desc="Epoch %4d, %s" % (0, loss_str))
        else:
            t = trange(
                1, self.oed.loss.num_steps + 1, desc="Epoch %4d, %s" % (0, loss_str), mininterval=60, maxinterval=600
            )

        for _ in t:
            loss = self.oed.step()
            loss = torch_item(loss)
            self.loss_history.append(loss)
            # Log every few losses -> too slow (and unnecessary to log everything)
            # the loss idx is added by 1 already when we call self.oed.step()
            if self.oed.loss.epoch_idx % 5 == 0 and self.oed.loss.step_per_epoch_idx == 0:
                loss_str = "Loss: {:3.3f} ".format(loss)
                t.set_description("Epoch %4d, %s" % (self.oed.loss.epoch_idx, loss_str))
            # Decrease LR at every new epoch
            if self.oed.loss.step_per_epoch_idx == 0:  # the loss idx is added by 1 already when we call self.oed.step()
                self.oed.optim.step()
                # evaluate in the end of each epoch
                epoch_size = (
                    self.oed.loss.step_idx
                    if self.oed.loss.epoch_idx == 1
                    else self.oed.loss.step_idx - self.check_point_overview.index[-1]
                )
                with torch.no_grad():
                    rmse_mean, rmse_stderr = self.oed.validation()
                self.check_point_overview.loc[self.oed.loss.step_idx] = [
                    loss,
                    np.mean(self.loss_history[-epoch_size:]),
                    torch_item(rmse_mean),
                    torch_item(rmse_stderr),
                ]
            # Log model every 20 epochs
            # the loss idx is added by 1 already when we call self.oed.step()
            if (
                self.oed.loss.epoch_idx > 0
                and self.oed.loss.epoch_idx % 20 == 0
                and self.oed.loss.step_per_epoch_idx == 0
                and self.oed.loss.step_idx < self.oed.loss.num_steps - 1
            ):
                self.save_checkpoint(self.oed.loss.step_idx)

        self.check_point_overview = self.check_point_overview.rename(index={self.oed.loss.step_idx: "final"})

        # evaluate and store results
        results = {
            "design_network": self.oed.process.design_net.cpu(),
            "seed": self.seed,
            "loss_history": self.loss_history,
            "loss_diff50": np.mean(self.loss_history[-51:-1]) / np.mean(self.loss_history[0:50]) - 1,
        }

        print("Training completed.")

        self.save_training_results()

        return results

    def save_checkpoint(self, idx: int):
        # Log model
        print(f"Storing checkpoint_{idx}... ", end="")
        # store the model:
        torch.save(self.oed.process.design_net.state_dict(), self.result_path / f"model_checkpoint_{idx}.pth")
        torch.save(self.oed.optim.get_state(), self.result_path / f"optimizer_scheduler_checkpoint_{idx}.pth")
        print(f"Checkpoint logged in {self.result_path}.")

    def save_training_results(self):
        # Log model
        print("Storing model... ", end="")
        # store the model:
        torch.save(self.oed.process.design_net.state_dict(), self.result_path / "model_checkpoint_final.pth")
        torch.save(self.oed.optim.get_state(), self.result_path / "optimizer_scheduler_checkpoint_final.pth")
        # store losses
        with pd.ExcelWriter(self.root_path / "check_point_overview.xlsx", mode="w") as writer:
            self.check_point_overview.to_excel(writer, sheet_name="training_overview")

        write_dict_to_json({"loss": self.loss_history}, self.result_path / "loss.json")
        write_dict_to_json(
            {"loss_first50": self.loss_history[:50], "loss_last50": self.loss_history[-50:]},
            self.result_path / "loss_50.json",
        )
        print(f"Model sotred in {self.result_path}.")
