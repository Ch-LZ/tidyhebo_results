
from tidyhebo.optimizer import gen_init_design


from tidyhebo.design_space import MixedDesignSpace
from tidyhebo.mixed_params import AbstractParameter
from tidyhebo.common import GA_Tensor, DS_Tensor, DS_df

from tidyhebo.base_model import SimpleModel

from typing import Generator, List
import torch

from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.optim import optimize_acqf


class LogEIMixedOptimizer:
    """
    A standard GP/EI setup without transforms, using a numerically stable acquisition function and a deliberately large Sobol sample count.
    """
    def __init__(self, parameters: List[AbstractParameter], train_x: DS_df | DS_Tensor=None, train_y: torch.Tensor=None, n_init: int | None = None):
        """
        n_init - number of initial-design points (preferably a power of two); None disables the initial design.
        """
        self.design_space: MixedDesignSpace = MixedDesignSpace(parameters)

        dim = len(parameters)

        self._init_design_generator: Generator[DS_df, None, None] = None
        if (n_init is not None) and (n_init > 0):
            self._init_design_generator = gen_init_design(n_init, self.design_space, return_df=True)

        train_x = torch.empty((0, dim), dtype=torch.double) if train_x is None else train_x
        self.train_x: GA_Tensor = self.design_space.from_DS_to_GA(train_x)

        train_y = torch.empty((0, 1), dtype=torch.double) if train_y is None else train_y
        self.train_y: torch.Tensor = train_y
        # model config
        self.kernel = "Matern52"
        self.power_outcome = False
        self._use_gate = False
        self.warp_input = False
        self.state_dict = None


    def suggest(self) -> DS_df:
        another_init_cand = next(self._init_design_generator, None)
        if another_init_cand is not None:
            return another_init_cand

        model: SimpleModel = SimpleModel(
            train_x=self.design_space.from_GA_to_GP(self.train_x),
            train_y=self.train_y,
            kernel=self.kernel,
            power_outcome=self.power_outcome,
            warp_input=self.warp_input,
            use_gate=self._use_gate,
            _best_train=False,
        )
        # update model
        model.train_model(state_dict=self.state_dict)
        # save to future runs
        self.state_dict = model.model.state_dict()
        # make acquisition function
        samplint_size = 512
        acqf = qLogExpectedImprovement(
            model=model.model,
            best_f=self.train_y.max().item(),
            sampler=SobolQMCNormalSampler(sample_shape=torch.Size([samplint_size]))
        )
        # optimizer acquisition
        candidates, _ = optimize_acqf(
            acq_function=acqf,
            bounds=model.bounds,
            q=1,
            num_restarts=10,
            raw_samples=min(1024, 500 * self.train_x.shape[-1]),
        )
        # transform into space
        res = self.design_space.from_GP_to_DS(candidates, return_df=True)
        return res


    def observe(self, X: DS_df | DS_Tensor, y: torch.Tensor) -> None:
        """
        Receive observations in design space.
        """
        # add new y observation to set
        if y.dim() == 1:
            y = y.reshape(-1, 1)  # column
        x = self.design_space.from_DS_to_GA(X)
        self.train_x = torch.cat([self.train_x, x], axis=0)
        self.train_y = torch.cat([self.train_y, y], axis=0)

