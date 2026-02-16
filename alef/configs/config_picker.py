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

from alef.configs.acquisition_function import (
    BasicRandomConfig,
    BasicSafeRandomConfig,
    BasicPredVarianceConfig,
    BasicPredEntropyConfig,
    BasicPredSigmaConfig,
    BasicSafePredEntropyConfig,
    BasicSafePredEntropyAllConfig,
    BasicSafeDiscoverConfig,
    BasicSafeDiscoverQuantileConfig,
    BasicSafeDiscoverEIConfig,
    BasicSafeDiscoverQuantileEIConfig,
    BasicSafeOptConfig,
    BasicSafeGPUCBConfig,
    BasicEIConfig,
    BasicSafeEIConfig,
    BasicSafeDiscoverOptConfig,
    BasicSafeDiscoverOptQuantileConfig,
)
from alef.configs.acquisition_function.bo_acquisition_functions.gp_ucb_config import BasicGPUCBConfig
from alef.configs.acquisition_function.bo_acquisition_functions.integrated_ei_config import BasicIntegratedEIConfig
from alef.configs.active_learner.active_learner_oracle_configs import (
    PredEntropyActiveLearnerOracleConfig,
    PredVarActiveLearnerOracleConfig,
    RandomActiveLearnerOracleConfig,
)
from alef.configs.active_learner.continuous_policy_active_learner_configs import (
    BasicContinuousPolicyActiveLearnerOracleConfig,
)
from alef.configs.bayesian_optimization.bayesian_optimizer_configs import (
    BOExpectedImprovementConfig,
    BOGPUCBConfig,
    BOIntegratedExpectedImprovementConfig,
)
from alef.configs.bayesian_optimization.bayesian_optimizer_objects_configs import (
    ObjectBOExpectedImprovementConfig,
    ObjectBOExpectedImprovementEAConfig,
    ObjectBOExpectedImprovementEAFewerStepsConfig,
    ObjectBOExpectedImprovementEAFlatWideConfig,
    ObjectBOExpectedImprovementPerSecondConfig,
    ObjectBOExpectedImprovementPerSecondEAConfig,
)
from alef.configs.bayesian_optimization.greedy_kernel_search_configs import (
    BaseGreedyKernelSearchConfig,
    GreedyKernelSearchBaseInitialConfig,
    GreedyKernelSearchNumNeighboursLimitedConfig,
)
from alef.configs.bayesian_optimization.treeGEP_optimizer_configs import (
    TreeGEPEvolutionaryOptimizerConfig,
    TreeGEPEvolutionaryOptimizerSmallPopulationConfig,
)
from alef.configs.kernels.grammar_tree_kernel_kernel_configs import (
    KernelGrammarSubtreeKernelConfig,
    OTWeightedDimsExtendedGrammarKernelConfig,
    OTWeightedDimsExtendedKernelWithHyperpriorConfig,
    OTWeightedDimsInvarianceGrammarKernelConfig,
    OptimalTransportGrammarKernelConfig,
    TreeBasedOTGrammarKernelConfig,
)
from alef.configs.kernels.hellinger_kernel_kernel_configs import (
    BasicHellingerKernelKernelConfig,
    HellingerKernelKernelSobolVirtualPoints,
)
from alef.configs.kernels.kernel_grammar_generators.cks_high_dim_generator_config import CKSHighDimGeneratorConfig
from alef.configs.kernels.kernel_grammar_generators.cks_with_rq_generator_config import (
    CKSWithRQGeneratorConfig,
    CKSWithRQTimeSeriesGeneratorConfig,
)
from alef.configs.kernels.kernel_grammar_generators.compositional_kernel_search_configs import (
    CKSTimeSeriesGeneratorConfig,
    CompositionalKernelSearchGeneratorConfig,
)
from alef.configs.kernels.kernel_grammar_generators.dynamic_hhk_generator_config import DynamicHHKGeneratorConfig
from alef.configs.kernels.kernel_grammar_generators.local_kernel_search_generator_config import (
    BigLocalNDimFullKernelsGrammarGeneratorConfig,
    FlatLocalKernelSearchSpaceConfig,
)
from alef.configs.kernels.kernel_grammar_generators.n_dim_full_kernels_generators_configs import (
    BigNDimFullKernelsGrammarGeneratorConfig,
    NDimFullKernelsGrammarGeneratorConfig,
)
from alef.configs.kernels.matern32_configs import BasicMatern32Config, Matern32WithPriorConfig
from alef.configs.kernels.rational_quadratic_configs import BasicRQConfig, RQWithPriorConfig
from alef.configs.kernels.spectral_mixture_kernel_config import BasicSMKernelConfig
from alef.configs.kernels.weighted_additive_kernel_config import (
    BasicWeightedAdditiveKernelConfig,
    WeightedAdditiveKernelWithPriorConfig,
)
from alef.configs.models.ahgp_model_config import AHGPModelConfig
from alef.configs.models.gp_model_amortized_ensemble_config import ExperimentalAmortizedEnsembleConfig
from alef.configs.models.gp_model_amortized_structured_config import ExperimentalAmortizedStructuredConfig
from alef.configs.models.gp_model_config import (
    BasicGPModelConfig,
    GPModelExtenseOptimization,
    GPModelFastConfig,
    GPModelFixedNoiseConfig,
    GPModelSmallPertubationConfig,
    GPModelWithNoisePriorConfig,
)
from alef.configs.models.gp_model_kernel_search_config import (
    GPFlatLocalKernelSearchConfig,
    GPKernelSearchCKSwithHighDim,
    GPKernelSearchCKSwithHighDimEvidence,
    GPKernelSearchCKSwithRQ,
    GPKernelSearchCKSwithRQEvidence,
)
from alef.configs.models.gp_model_marginalized_config import (
    BasicGPModelMarginalizedConfig,
    GPModelMarginalizedConfigMoreThinningConfig,
    GPModelMarginalizedConfigMoreSamplesConfig,
    GPModelMarginalizedConfigMoreSamplesMoreThinningConfig,
)
from alef.configs.models.gp_model_marginalized_config import (
    GPModelMarginalizedConfigMAPInitialized,
    GPModelMarginalizedConfigFast,
)
from alef.configs.models.gp_model_laplace_config import BasicGPModelLaplaceConfig
from alef.configs.kernels.hhk_configs import (
    HHKEightLocalDefaultConfig,
    HHKFourLocalDefaultConfig,
    HHKTwoLocalDefaultConfig,
)
from alef.configs.kernels.wami_configs import BasicWamiConfig
from alef.configs.kernels.rbf_configs import BasicRBFConfig, RBFWithPriorConfig
from alef.configs.models.svgp_model_config import BasicSVGPConfig
from alef.configs.active_learner.active_learner_configs import (
    PredEntropyActiveLearnerConfig,
    PredVarActiveLearnerConfig,
    RandomActiveLearnerConfig,
)
from alef.configs.models.mogp_model_config import BasicMOGPModelConfig
from alef.configs.active_learner.batch_active_learner_configs import (
    EntropyBatchActiveLearnerConfig,
    RandomBatchActiveLearnerConfig,
)
from alef.configs.kernels.additive_kernel_configs import BasicAdditiveKernelConfig, AdditiveKernelWithPriorConfig
from alef.configs.kernels.matern52_configs import BasicMatern52Config, Matern52WithPriorConfig
from alef.configs.kernels.linear_configs import BasicLinearConfig, LinearWithPriorConfig
from alef.configs.kernels.neural_kernel_network_config import BasicNKNConfig
from alef.configs.kernels.multi_output_kernels.coregionalization_1latent_kernel_configs import (
    BasicCoregionalization1LConfig,
    Coregionalization1LWithPriorConfig,
)
from alef.configs.kernels.multi_output_kernels.coregionalization_Platent_kernel_configs import (
    BasicCoregionalizationPLConfig,
    CoregionalizationPLWithPriorConfig,
)
from alef.configs.kernels.multi_output_kernels.coregionalization_kernel_configs import (
    BasicCoregionalizationSOConfig,
    CoregionalizationSOWithPriorConfig,
    BasicCoregionalizationMOConfig,
    CoregionalizationMOWithPriorConfig,
)
from alef.configs.kernels.multi_output_kernels.multi_source_additive_kernel_configs import (
    BasicMIAdditiveConfig,
    MIAdditiveWithPriorConfig,
)
from alef.configs.kernels.multi_output_kernels.flexible_transfer_kernel_config import (
    BasicFlexibleTransferConfig,
    FlexibleTransferWithPriorConfig,
)
from alef.configs.kernels.multi_output_kernels.coregionalization_transfer_kernel_config import (
    BasicCoregionalizationTransferConfig,
    CoregionalizationTransferWithPriorConfig,
)
from alef.configs.kernels.multi_output_kernels.fpacoh_kernel_config import BasicFPACOHKernelConfig
from alef.configs.models.metagp_model_config import BasicMetaGPModelConfig
from alef.configs.models.sparse_gp_model_config import (
    BasicSparseGPModelConfig,
    SparseGPModelFastConfig,
    SparseGPModelFixedNoiseConfig,
    SparseGPModel300IPConfig,
    SparseGPModel500IPConfig,
    SparseGPModel700IPConfig,
    SparseGPModel700IPExtenseConfig,
)
from alef.configs.models.deep_gp_config import DeepGPConfig, FiveLayerDeepGPConfig, ThreeLayerDeepGPConfig
from alef.configs.models.gp_model_scalable_config import (
    BasicScalableGPModelConfig,
    GPRAdamConfig,
    GPRAdamWithValidationSet,
    GPRAdamWithValidationSetNLL,
)
from alef.configs.models.mogp_model_so_config import BasicSOMOGPModelConfig
from alef.configs.models.mogp_model_transfer_config import BasicTransferGPModelConfig
from alef.configs.models.mogp_model_so_marginalized_config import (
    BasicSOMOGPModelMarginalizedConfig,
    SOMOGPModelMarginalizedConfigMoreThinningConfig,
    SOMOGPModelMarginalizedConfigMoreSamplesConfig,
    SOMOGPModelMarginalizedConfigMoreSamplesMoreThinningConfig,
    SOMOGPModelMarginalizedConfigMAPInitialized,
    SOMOGPModelMarginalizedConfigFast,
)
from alef.configs.models.gp_model_for_engine1_config import (
    Engine1GPModelBEConfig,
    Engine1GPModelTExConfig,
    Engine1GPModelPI0vConfig,
    Engine1GPModelPI0sConfig,
    Engine1GPModelHCConfig,
    Engine1GPModelNOxConfig,
)
from alef.configs.models.gp_model_for_engine2_config import (
    Engine2GPModelBEConfig,
    Engine2GPModelTExConfig,
    Engine2GPModelPI0vConfig,
    Engine2GPModelPI0sConfig,
    Engine2GPModelHCConfig,
    Engine2GPModelNOxConfig,
)
from alef.configs.kernels.deep_kernels.invertible_resnet_kernel_configs import (
    BasicInvertibleResnetKernelConfig,
    ExploreRegularizedIResnetKernelConfig,
    InvertibleResnetKernelWithPriorConfig,
    CurlRegularizedIResnetKernelConfig,
    AxisRegularizedIResnetKernelConfig,
    InvertibleResnetWithLayerNoiseKernelConfig,
)
from alef.configs.kernels.deep_kernels.mlp_deep_kernel_config import (
    BasicMLPDeepKernelConfig,
    MLPWithPriorDeepKernelConfig,
    SmallMLPWithPriorDeepKernelConfig,
)
from alef.configs.kernels.kernel_list_configs import (
    SEKernelViaKernelListConfig,
    PERKernelViaKernelListConfig,
    ExperimentalKernelListConfig,
)
from alef.configs.experiment.simulator_configs.base_simulator_config import BaseSimulatorConfig
from alef.configs.experiment.simulator_configs.single_task_1d_illustrate_config import SingleTaskIllustrateConfig
from alef.configs.experiment.simulator_configs.transfer_task_1d_illustrate_config import TransferTaskIllustrateConfig
from alef.configs.experiment.simulator_configs.single_task_branin_config import (
    SingleTaskBraninConfig,
    SingleTaskBranin0Config,
    SingleTaskBranin1Config,
    SingleTaskBranin2Config,
    SingleTaskBranin3Config,
    SingleTaskBranin4Config,
)
from alef.configs.experiment.simulator_configs.transfer_task_branin_config import (
    TransferTaskBraninBaseConfig,
    TransferTaskBranin0Config,
    TransferTaskBranin1Config,
    TransferTaskBranin2Config,
    TransferTaskBranin3Config,
    TransferTaskBranin4Config,
)
from alef.configs.experiment.simulator_configs.single_task_mogp1d_config import (
    SingleTaskMOGP1DBaseConfig,
    SingleTaskMOGP1D0Config,
    SingleTaskMOGP1D1Config,
    SingleTaskMOGP1D2Config,
    SingleTaskMOGP1D3Config,
    SingleTaskMOGP1D4Config,
    SingleTaskMOGP1D5Config,
    SingleTaskMOGP1D6Config,
    SingleTaskMOGP1D7Config,
    SingleTaskMOGP1D8Config,
    SingleTaskMOGP1D9Config,
    SingleTaskMOGP1D10Config,
    SingleTaskMOGP1D11Config,
    SingleTaskMOGP1D12Config,
    SingleTaskMOGP1D13Config,
    SingleTaskMOGP1D14Config,
    SingleTaskMOGP1D15Config,
    SingleTaskMOGP1D16Config,
    SingleTaskMOGP1D17Config,
    SingleTaskMOGP1D18Config,
    SingleTaskMOGP1D19Config,
)
from alef.configs.experiment.simulator_configs.transfer_task_mogp1d_config import (
    TransferTaskMOGP1DBaseConfig,
    TransferTaskMOGP1D0Config,
    TransferTaskMOGP1D1Config,
    TransferTaskMOGP1D2Config,
    TransferTaskMOGP1D3Config,
    TransferTaskMOGP1D4Config,
    TransferTaskMOGP1D5Config,
    TransferTaskMOGP1D6Config,
    TransferTaskMOGP1D7Config,
    TransferTaskMOGP1D8Config,
    TransferTaskMOGP1D9Config,
    TransferTaskMOGP1D10Config,
    TransferTaskMOGP1D11Config,
    TransferTaskMOGP1D12Config,
    TransferTaskMOGP1D13Config,
    TransferTaskMOGP1D14Config,
    TransferTaskMOGP1D15Config,
    TransferTaskMOGP1D16Config,
    TransferTaskMOGP1D17Config,
    TransferTaskMOGP1D18Config,
    TransferTaskMOGP1D19Config,
)
from alef.configs.experiment.simulator_configs.single_task_mogp1dz_config import (
    SingleTaskMOGP1DzBaseConfig,
    SingleTaskMOGP1Dz0Config,
    SingleTaskMOGP1Dz1Config,
    SingleTaskMOGP1Dz2Config,
    SingleTaskMOGP1Dz3Config,
    SingleTaskMOGP1Dz4Config,
    SingleTaskMOGP1Dz5Config,
    SingleTaskMOGP1Dz6Config,
    SingleTaskMOGP1Dz7Config,
    SingleTaskMOGP1Dz8Config,
    SingleTaskMOGP1Dz9Config,
    SingleTaskMOGP1Dz10Config,
    SingleTaskMOGP1Dz11Config,
    SingleTaskMOGP1Dz12Config,
    SingleTaskMOGP1Dz13Config,
    SingleTaskMOGP1Dz14Config,
    SingleTaskMOGP1Dz15Config,
    SingleTaskMOGP1Dz16Config,
    SingleTaskMOGP1Dz17Config,
    SingleTaskMOGP1Dz18Config,
    SingleTaskMOGP1Dz19Config,
)
from alef.configs.experiment.simulator_configs.transfer_task_mogp1dz_config import (
    TransferTaskMOGP1DzBaseConfig,
    TransferTaskMOGP1Dz0Config,
    TransferTaskMOGP1Dz1Config,
    TransferTaskMOGP1Dz2Config,
    TransferTaskMOGP1Dz3Config,
    TransferTaskMOGP1Dz4Config,
    TransferTaskMOGP1Dz5Config,
    TransferTaskMOGP1Dz6Config,
    TransferTaskMOGP1Dz7Config,
    TransferTaskMOGP1Dz8Config,
    TransferTaskMOGP1Dz9Config,
    TransferTaskMOGP1Dz10Config,
    TransferTaskMOGP1Dz11Config,
    TransferTaskMOGP1Dz12Config,
    TransferTaskMOGP1Dz13Config,
    TransferTaskMOGP1Dz14Config,
    TransferTaskMOGP1Dz15Config,
    TransferTaskMOGP1Dz16Config,
    TransferTaskMOGP1Dz17Config,
    TransferTaskMOGP1Dz18Config,
    TransferTaskMOGP1Dz19Config,
)

from alef.configs.experiment.simulator_configs.single_task_mogp2d_config import (
    SingleTaskMOGP2DBaseConfig,
    SingleTaskMOGP2D0Config,
    SingleTaskMOGP2D1Config,
    SingleTaskMOGP2D2Config,
    SingleTaskMOGP2D3Config,
    SingleTaskMOGP2D4Config,
    SingleTaskMOGP2D5Config,
    SingleTaskMOGP2D6Config,
    SingleTaskMOGP2D7Config,
    SingleTaskMOGP2D8Config,
    SingleTaskMOGP2D9Config,
    SingleTaskMOGP2D10Config,
    SingleTaskMOGP2D11Config,
    SingleTaskMOGP2D12Config,
    SingleTaskMOGP2D13Config,
    SingleTaskMOGP2D14Config,
    SingleTaskMOGP2D15Config,
    SingleTaskMOGP2D16Config,
    SingleTaskMOGP2D17Config,
    SingleTaskMOGP2D18Config,
    SingleTaskMOGP2D19Config,
)
from alef.configs.experiment.simulator_configs.transfer_task_mogp2d_config import (
    TransferTaskMOGP2DBaseConfig,
    TransferTaskMOGP2D0Config,
    TransferTaskMOGP2D1Config,
    TransferTaskMOGP2D2Config,
    TransferTaskMOGP2D3Config,
    TransferTaskMOGP2D4Config,
    TransferTaskMOGP2D5Config,
    TransferTaskMOGP2D6Config,
    TransferTaskMOGP2D7Config,
    TransferTaskMOGP2D8Config,
    TransferTaskMOGP2D9Config,
    TransferTaskMOGP2D10Config,
    TransferTaskMOGP2D11Config,
    TransferTaskMOGP2D12Config,
    TransferTaskMOGP2D13Config,
    TransferTaskMOGP2D14Config,
    TransferTaskMOGP2D15Config,
    TransferTaskMOGP2D16Config,
    TransferTaskMOGP2D17Config,
    TransferTaskMOGP2D18Config,
    TransferTaskMOGP2D19Config,
)
from alef.configs.experiment.simulator_configs.single_task_mogp2dz_config import (
    SingleTaskMOGP2DzBaseConfig,
    SingleTaskMOGP2Dz0Config,
    SingleTaskMOGP2Dz1Config,
    SingleTaskMOGP2Dz2Config,
    SingleTaskMOGP2Dz3Config,
    SingleTaskMOGP2Dz4Config,
    SingleTaskMOGP2Dz5Config,
    SingleTaskMOGP2Dz6Config,
    SingleTaskMOGP2Dz7Config,
    SingleTaskMOGP2Dz8Config,
    SingleTaskMOGP2Dz9Config,
    SingleTaskMOGP2Dz10Config,
    SingleTaskMOGP2Dz11Config,
    SingleTaskMOGP2Dz12Config,
    SingleTaskMOGP2Dz13Config,
    SingleTaskMOGP2Dz14Config,
    SingleTaskMOGP2Dz15Config,
    SingleTaskMOGP2Dz16Config,
    SingleTaskMOGP2Dz17Config,
    SingleTaskMOGP2Dz18Config,
    SingleTaskMOGP2Dz19Config,
)
from alef.configs.experiment.simulator_configs.transfer_task_mogp2dz_config import (
    TransferTaskMOGP2DzBaseConfig,
    TransferTaskMOGP2Dz0Config,
    TransferTaskMOGP2Dz1Config,
    TransferTaskMOGP2Dz2Config,
    TransferTaskMOGP2Dz3Config,
    TransferTaskMOGP2Dz4Config,
    TransferTaskMOGP2Dz5Config,
    TransferTaskMOGP2Dz6Config,
    TransferTaskMOGP2Dz7Config,
    TransferTaskMOGP2Dz8Config,
    TransferTaskMOGP2Dz9Config,
    TransferTaskMOGP2Dz10Config,
    TransferTaskMOGP2Dz11Config,
    TransferTaskMOGP2Dz12Config,
    TransferTaskMOGP2Dz13Config,
    TransferTaskMOGP2Dz14Config,
    TransferTaskMOGP2Dz15Config,
    TransferTaskMOGP2Dz16Config,
    TransferTaskMOGP2Dz17Config,
    TransferTaskMOGP2Dz18Config,
    TransferTaskMOGP2Dz19Config,
)
from alef.configs.experiment.simulator_configs.single_task_engine_interpolated_config import (
    SingleTaskEngineInterpolatedBaseConfig,
    SingleTaskEngineInterpolated_be_Config,
    SingleTaskEngineInterpolated_TEx_Config,
    SingleTaskEngineInterpolated_PI0v_Config,
    SingleTaskEngineInterpolated_PI0s_Config,
    SingleTaskEngineInterpolated_HC_Config,
    SingleTaskEngineInterpolated_NOx_Config,
)
from alef.configs.experiment.simulator_configs.transfer_task_engine_interpolated_config import (
    TransferTaskEngineInterpolatedBaseConfig,
    TransferTaskEngineInterpolated_be_Config,
    TransferTaskEngineInterpolated_TEx_Config,
    TransferTaskEngineInterpolated_PI0v_Config,
    TransferTaskEngineInterpolated_PI0s_Config,
    TransferTaskEngineInterpolated_HC_Config,
    TransferTaskEngineInterpolated_NOx_Config,
)


class ConfigPicker:
    models_configs_dict = {
        c.__name__: c
        for c in [
            BasicGPModelConfig,
            GPModelFastConfig,
            GPModelWithNoisePriorConfig,
            GPModelSmallPertubationConfig,
            GPModelExtenseOptimization,
            GPModelFixedNoiseConfig,
            BasicGPModelMarginalizedConfig,
            GPModelMarginalizedConfigMoreThinningConfig,
            GPModelMarginalizedConfigMoreSamplesConfig,
            GPModelMarginalizedConfigMoreSamplesMoreThinningConfig,
            BasicGPModelLaplaceConfig,
            GPModelMarginalizedConfigMAPInitialized,
            GPModelMarginalizedConfigFast,
            BasicSVGPConfig,
            BasicMOGPModelConfig,
            BasicSparseGPModelConfig,
            SparseGPModelFastConfig,
            SparseGPModelFixedNoiseConfig,
            SparseGPModel300IPConfig,
            SparseGPModel500IPConfig,
            SparseGPModel700IPConfig,
            SparseGPModel700IPExtenseConfig,
            DeepGPConfig,
            ThreeLayerDeepGPConfig,
            FiveLayerDeepGPConfig,
            BasicScalableGPModelConfig,
            GPRAdamConfig,
            GPRAdamWithValidationSet,
            GPRAdamWithValidationSetNLL,
            GPKernelSearchCKSwithHighDim,
            GPKernelSearchCKSwithRQ,
            GPKernelSearchCKSwithRQEvidence,
            GPKernelSearchCKSwithHighDimEvidence,
            GPFlatLocalKernelSearchConfig,
            AHGPModelConfig,
            BasicMOGPModelConfig,
            BasicSOMOGPModelConfig,
            BasicTransferGPModelConfig,
            BasicSOMOGPModelMarginalizedConfig,
            SOMOGPModelMarginalizedConfigMoreThinningConfig,
            SOMOGPModelMarginalizedConfigMoreSamplesConfig,
            SOMOGPModelMarginalizedConfigMoreSamplesMoreThinningConfig,
            SOMOGPModelMarginalizedConfigMAPInitialized,
            SOMOGPModelMarginalizedConfigFast,
            BasicMetaGPModelConfig,
            Engine1GPModelBEConfig,
            Engine1GPModelTExConfig,
            Engine1GPModelPI0vConfig,
            Engine1GPModelPI0sConfig,
            Engine1GPModelHCConfig,
            Engine1GPModelNOxConfig,
            Engine2GPModelBEConfig,
            Engine2GPModelTExConfig,
            Engine2GPModelPI0vConfig,
            Engine2GPModelPI0sConfig,
            Engine2GPModelHCConfig,
            Engine2GPModelNOxConfig,
            ExperimentalAmortizedStructuredConfig,
            ExperimentalAmortizedEnsembleConfig,
        ]
    }

    kernels_configs_dict = {
        c.__name__: c
        for c in [
            HHKEightLocalDefaultConfig,
            HHKFourLocalDefaultConfig,
            HHKTwoLocalDefaultConfig,
            BasicWamiConfig,
            RBFWithPriorConfig,
            BasicRBFConfig,
            BasicAdditiveKernelConfig,
            AdditiveKernelWithPriorConfig,
            BasicMatern52Config,
            Matern52WithPriorConfig,
            BasicMatern32Config,
            Matern32WithPriorConfig,
            BasicLinearConfig,
            LinearWithPriorConfig,
            BasicRQConfig,
            RQWithPriorConfig,
            BasicNKNConfig,
            BasicMLPDeepKernelConfig,
            MLPWithPriorDeepKernelConfig,
            SmallMLPWithPriorDeepKernelConfig,
            BasicInvertibleResnetKernelConfig,
            InvertibleResnetKernelWithPriorConfig,
            InvertibleResnetWithLayerNoiseKernelConfig,
            CurlRegularizedIResnetKernelConfig,
            AxisRegularizedIResnetKernelConfig,
            KernelGrammarSubtreeKernelConfig,
            ExploreRegularizedIResnetKernelConfig,
            BasicHellingerKernelKernelConfig,
            HellingerKernelKernelSobolVirtualPoints,
            OptimalTransportGrammarKernelConfig,
            TreeBasedOTGrammarKernelConfig,
            OTWeightedDimsExtendedGrammarKernelConfig,
            OTWeightedDimsInvarianceGrammarKernelConfig,
            OTWeightedDimsExtendedKernelWithHyperpriorConfig,
            BasicWeightedAdditiveKernelConfig,
            WeightedAdditiveKernelWithPriorConfig,  #
            BasicSMKernelConfig,
            BasicCoregionalization1LConfig,
            Coregionalization1LWithPriorConfig,
            BasicCoregionalizationPLConfig,
            CoregionalizationPLWithPriorConfig,
            BasicCoregionalizationSOConfig,
            CoregionalizationSOWithPriorConfig,
            BasicCoregionalizationMOConfig,
            CoregionalizationMOWithPriorConfig,
            BasicMIAdditiveConfig,
            MIAdditiveWithPriorConfig,
            BasicFlexibleTransferConfig,
            FlexibleTransferWithPriorConfig,
            BasicCoregionalizationTransferConfig,
            CoregionalizationTransferWithPriorConfig,
            BasicFPACOHKernelConfig,
            SEKernelViaKernelListConfig,
            PERKernelViaKernelListConfig,
            ExperimentalKernelListConfig,
        ]
    }

    acquisition_function_configs_dict = {
        c.__name__: c
        for c in [
            BasicRandomConfig,
            BasicSafeRandomConfig,
            BasicPredVarianceConfig,
            BasicPredSigmaConfig,
            BasicPredEntropyConfig,
            BasicSafePredEntropyConfig,
            BasicSafePredEntropyAllConfig,
            BasicSafeDiscoverConfig,
            BasicSafeDiscoverQuantileConfig,
            BasicSafeDiscoverEIConfig,
            BasicSafeDiscoverQuantileEIConfig,
            BasicSafeOptConfig,
            BasicSafeGPUCBConfig,
            BasicEIConfig,
            BasicGPUCBConfig,
            BasicIntegratedEIConfig,
            BasicSafeEIConfig,
            BasicSafeDiscoverOptConfig,
            BasicSafeDiscoverOptQuantileConfig,
        ]
    }

    active_learner_configs_dict = {
        c.__name__: c
        for c in [
            PredEntropyActiveLearnerConfig,
            PredVarActiveLearnerConfig,
            RandomActiveLearnerConfig,
            PredVarActiveLearnerOracleConfig,
            PredEntropyActiveLearnerOracleConfig,
            RandomActiveLearnerOracleConfig,
            BasicContinuousPolicyActiveLearnerOracleConfig,
        ]
    }

    batch_active_learner_configs_dict = {
        c.__name__: c for c in [EntropyBatchActiveLearnerConfig, RandomBatchActiveLearnerConfig]
    }

    bayesian_optimization_configs_dict = {
        c.__name__: c
        for c in [
            BOExpectedImprovementConfig,
            BOGPUCBConfig,
            BOIntegratedExpectedImprovementConfig,
            ObjectBOExpectedImprovementConfig,
            ObjectBOExpectedImprovementEAConfig,
            ObjectBOExpectedImprovementEAFewerStepsConfig,
            ObjectBOExpectedImprovementEAFlatWideConfig,
            ObjectBOExpectedImprovementPerSecondEAConfig,
            ObjectBOExpectedImprovementPerSecondConfig,
        ]
    }

    greedy_kernel_seach_configs_dict = {
        c.__name__: c
        for c in [
            GreedyKernelSearchNumNeighboursLimitedConfig,
            BaseGreedyKernelSearchConfig,
            TreeGEPEvolutionaryOptimizerConfig,
            TreeGEPEvolutionaryOptimizerSmallPopulationConfig,
            GreedyKernelSearchBaseInitialConfig,
        ]
    }

    kernel_grammar_generator_configs_dict = {
        c.__name__: c
        for c in [
            NDimFullKernelsGrammarGeneratorConfig,
            CompositionalKernelSearchGeneratorConfig,
            CKSTimeSeriesGeneratorConfig,
            CKSWithRQGeneratorConfig,
            CKSWithRQTimeSeriesGeneratorConfig,
            CKSHighDimGeneratorConfig,
            DynamicHHKGeneratorConfig,
            FlatLocalKernelSearchSpaceConfig,
            BigLocalNDimFullKernelsGrammarGeneratorConfig,
            BigNDimFullKernelsGrammarGeneratorConfig,
        ]
    }

    experiment_simulator_configs_dict = {
        c.__name__: c
        for c in [
            BaseSimulatorConfig,
            SingleTaskIllustrateConfig,
            TransferTaskIllustrateConfig,
            SingleTaskBraninConfig,
            SingleTaskBranin0Config,
            SingleTaskBranin1Config,
            SingleTaskBranin2Config,
            SingleTaskBranin3Config,
            SingleTaskBranin4Config,
            TransferTaskBraninBaseConfig,
            TransferTaskBranin0Config,
            TransferTaskBranin1Config,
            TransferTaskBranin2Config,
            TransferTaskBranin3Config,
            TransferTaskBranin4Config,
            SingleTaskMOGP1DBaseConfig,
            SingleTaskMOGP1D0Config,
            SingleTaskMOGP1D1Config,
            SingleTaskMOGP1D2Config,
            SingleTaskMOGP1D3Config,
            SingleTaskMOGP1D4Config,
            SingleTaskMOGP1D5Config,
            SingleTaskMOGP1D6Config,
            SingleTaskMOGP1D7Config,
            SingleTaskMOGP1D8Config,
            SingleTaskMOGP1D9Config,
            SingleTaskMOGP1D10Config,
            SingleTaskMOGP1D11Config,
            SingleTaskMOGP1D12Config,
            SingleTaskMOGP1D13Config,
            SingleTaskMOGP1D14Config,
            SingleTaskMOGP1D15Config,
            SingleTaskMOGP1D16Config,
            SingleTaskMOGP1D17Config,
            SingleTaskMOGP1D18Config,
            SingleTaskMOGP1D19Config,
            TransferTaskMOGP1DBaseConfig,
            TransferTaskMOGP1D0Config,
            TransferTaskMOGP1D1Config,
            TransferTaskMOGP1D2Config,
            TransferTaskMOGP1D3Config,
            TransferTaskMOGP1D4Config,
            TransferTaskMOGP1D5Config,
            TransferTaskMOGP1D6Config,
            TransferTaskMOGP1D7Config,
            TransferTaskMOGP1D8Config,
            TransferTaskMOGP1D9Config,
            TransferTaskMOGP1D10Config,
            TransferTaskMOGP1D11Config,
            TransferTaskMOGP1D12Config,
            TransferTaskMOGP1D13Config,
            TransferTaskMOGP1D14Config,
            TransferTaskMOGP1D15Config,
            TransferTaskMOGP1D16Config,
            TransferTaskMOGP1D17Config,
            TransferTaskMOGP1D18Config,
            TransferTaskMOGP1D19Config,
            SingleTaskMOGP1DzBaseConfig,
            SingleTaskMOGP1Dz0Config,
            SingleTaskMOGP1Dz1Config,
            SingleTaskMOGP1Dz2Config,
            SingleTaskMOGP1Dz3Config,
            SingleTaskMOGP1Dz4Config,
            SingleTaskMOGP1Dz5Config,
            SingleTaskMOGP1Dz6Config,
            SingleTaskMOGP1Dz7Config,
            SingleTaskMOGP1Dz8Config,
            SingleTaskMOGP1Dz9Config,
            SingleTaskMOGP1Dz10Config,
            SingleTaskMOGP1Dz11Config,
            SingleTaskMOGP1Dz12Config,
            SingleTaskMOGP1Dz13Config,
            SingleTaskMOGP1Dz14Config,
            SingleTaskMOGP1Dz15Config,
            SingleTaskMOGP1Dz16Config,
            SingleTaskMOGP1Dz17Config,
            SingleTaskMOGP1Dz18Config,
            SingleTaskMOGP1Dz19Config,
            TransferTaskMOGP1DzBaseConfig,
            TransferTaskMOGP1Dz0Config,
            TransferTaskMOGP1Dz1Config,
            TransferTaskMOGP1Dz2Config,
            TransferTaskMOGP1Dz3Config,
            TransferTaskMOGP1Dz4Config,
            TransferTaskMOGP1Dz5Config,
            TransferTaskMOGP1Dz6Config,
            TransferTaskMOGP1Dz7Config,
            TransferTaskMOGP1Dz8Config,
            TransferTaskMOGP1Dz9Config,
            TransferTaskMOGP1Dz10Config,
            TransferTaskMOGP1Dz11Config,
            TransferTaskMOGP1Dz12Config,
            TransferTaskMOGP1Dz13Config,
            TransferTaskMOGP1Dz14Config,
            TransferTaskMOGP1Dz15Config,
            TransferTaskMOGP1Dz16Config,
            TransferTaskMOGP1Dz17Config,
            TransferTaskMOGP1Dz18Config,
            TransferTaskMOGP1Dz19Config,
            SingleTaskMOGP2DBaseConfig,
            SingleTaskMOGP2D0Config,
            SingleTaskMOGP2D1Config,
            SingleTaskMOGP2D2Config,
            SingleTaskMOGP2D3Config,
            SingleTaskMOGP2D4Config,
            SingleTaskMOGP2D5Config,
            SingleTaskMOGP2D6Config,
            SingleTaskMOGP2D7Config,
            SingleTaskMOGP2D8Config,
            SingleTaskMOGP2D9Config,
            SingleTaskMOGP2D10Config,
            SingleTaskMOGP2D11Config,
            SingleTaskMOGP2D12Config,
            SingleTaskMOGP2D13Config,
            SingleTaskMOGP2D14Config,
            SingleTaskMOGP2D15Config,
            SingleTaskMOGP2D16Config,
            SingleTaskMOGP2D17Config,
            SingleTaskMOGP2D18Config,
            SingleTaskMOGP2D19Config,
            TransferTaskMOGP2DBaseConfig,
            TransferTaskMOGP2D0Config,
            TransferTaskMOGP2D1Config,
            TransferTaskMOGP2D2Config,
            TransferTaskMOGP2D3Config,
            TransferTaskMOGP2D4Config,
            TransferTaskMOGP2D5Config,
            TransferTaskMOGP2D6Config,
            TransferTaskMOGP2D7Config,
            TransferTaskMOGP2D8Config,
            TransferTaskMOGP2D9Config,
            TransferTaskMOGP2D10Config,
            TransferTaskMOGP2D11Config,
            TransferTaskMOGP2D12Config,
            TransferTaskMOGP2D13Config,
            TransferTaskMOGP2D14Config,
            TransferTaskMOGP2D15Config,
            TransferTaskMOGP2D16Config,
            TransferTaskMOGP2D17Config,
            TransferTaskMOGP2D18Config,
            TransferTaskMOGP2D19Config,
            SingleTaskMOGP2DzBaseConfig,
            SingleTaskMOGP2Dz0Config,
            SingleTaskMOGP2Dz1Config,
            SingleTaskMOGP2Dz2Config,
            SingleTaskMOGP2Dz3Config,
            SingleTaskMOGP2Dz4Config,
            SingleTaskMOGP2Dz5Config,
            SingleTaskMOGP2Dz6Config,
            SingleTaskMOGP2Dz7Config,
            SingleTaskMOGP2Dz8Config,
            SingleTaskMOGP2Dz9Config,
            SingleTaskMOGP2Dz10Config,
            SingleTaskMOGP2Dz11Config,
            SingleTaskMOGP2Dz12Config,
            SingleTaskMOGP2Dz13Config,
            SingleTaskMOGP2Dz14Config,
            SingleTaskMOGP2Dz15Config,
            SingleTaskMOGP2Dz16Config,
            SingleTaskMOGP2Dz17Config,
            SingleTaskMOGP2Dz18Config,
            SingleTaskMOGP2Dz19Config,
            TransferTaskMOGP2DzBaseConfig,
            TransferTaskMOGP2Dz0Config,
            TransferTaskMOGP2Dz1Config,
            TransferTaskMOGP2Dz2Config,
            TransferTaskMOGP2Dz3Config,
            TransferTaskMOGP2Dz4Config,
            TransferTaskMOGP2Dz5Config,
            TransferTaskMOGP2Dz6Config,
            TransferTaskMOGP2Dz7Config,
            TransferTaskMOGP2Dz8Config,
            TransferTaskMOGP2Dz9Config,
            TransferTaskMOGP2Dz10Config,
            TransferTaskMOGP2Dz11Config,
            TransferTaskMOGP2Dz12Config,
            TransferTaskMOGP2Dz13Config,
            TransferTaskMOGP2Dz14Config,
            TransferTaskMOGP2Dz15Config,
            TransferTaskMOGP2Dz16Config,
            TransferTaskMOGP2Dz17Config,
            TransferTaskMOGP2Dz18Config,
            TransferTaskMOGP2Dz19Config,
            SingleTaskEngineInterpolatedBaseConfig,
            SingleTaskEngineInterpolated_be_Config,
            SingleTaskEngineInterpolated_TEx_Config,
            SingleTaskEngineInterpolated_PI0v_Config,
            SingleTaskEngineInterpolated_PI0s_Config,
            SingleTaskEngineInterpolated_HC_Config,
            SingleTaskEngineInterpolated_NOx_Config,
            TransferTaskEngineInterpolatedBaseConfig,
            TransferTaskEngineInterpolated_be_Config,
            TransferTaskEngineInterpolated_TEx_Config,
            TransferTaskEngineInterpolated_PI0v_Config,
            TransferTaskEngineInterpolated_PI0s_Config,
            TransferTaskEngineInterpolated_HC_Config,
            TransferTaskEngineInterpolated_NOx_Config,
        ]
    }

    @staticmethod
    def pick_kernel_config(config_class_name):
        return ConfigPicker.kernels_configs_dict[config_class_name]

    @staticmethod
    def pick_model_config(config_class_name):
        return ConfigPicker.models_configs_dict[config_class_name]

    @staticmethod
    def pick_acquisition_function_config(config_class_name):
        return ConfigPicker.acquisition_function_configs_dict[config_class_name]

    @staticmethod
    def pick_active_learner_config(config_class_name):
        return ConfigPicker.active_learner_configs_dict[config_class_name]

    @staticmethod
    def pick_batch_active_learner_config(config_class_name):
        return ConfigPicker.batch_active_learner_configs_dict[config_class_name]

    @staticmethod
    def pick_bayesian_optimization_config(config_class_name):
        return ConfigPicker.bayesian_optimization_configs_dict[config_class_name]

    @staticmethod
    def pick_kernel_grammar_generator_config(config_class_name):
        return ConfigPicker.kernel_grammar_generator_configs_dict[config_class_name]

    @staticmethod
    def pick_greedy_kernel_search_config(config_class_name):
        return ConfigPicker.greedy_kernel_seach_configs_dict[config_class_name]

    @staticmethod
    def pick_experiment_simulator_config(config_class_name):
        return ConfigPicker.experiment_simulator_configs_dict[config_class_name]
