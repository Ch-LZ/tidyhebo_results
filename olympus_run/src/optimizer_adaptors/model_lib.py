import abc
import importlib
import sys
from pathlib import Path
from typing import Dict, Type, Union, Any, NewType, Optional, List
import numpy as np
import yaml

from .simplex_transformer import SimplexTransformer

if sys.version_info < (3, 7):
    sys.exit("Python 3.7 or higher is required.")

# region point types
XPoint = NewType('XPoint', np.ndarray)
YPoint = NewType('YPoint', np.ndarray)

def make_x_point(data: np.ndarray) -> XPoint:
    # Allow any number of rows (>=1) to support batches.
    if data.ndim != 2 or data.shape[0] < 1:
        raise ValueError("XPoint must have shape (N, -1) with N >= 1")
    return XPoint(data)

def make_y_point(data: np.ndarray) -> YPoint:
    if data.ndim != 2 or data.shape[1] != 1:
        raise ValueError("YPoint must have shape (-1, 1)")
    return YPoint(data)
# endregion

# region model

class Model(abc.ABC):
    """Base class for all optimizers."""
    # Fully qualified adaptor class paths.
    _MODEL_REGISTRY: Dict[str, str] = {
        'optuna': 'src.optimizer_adaptors.optuna_opt.OptunaOptimizer',
        'random': 'src.optimizer_adaptors.random_opt.RandomOptimizer',
        'logei': 'src.optimizer_adaptors.logei_opt.LogEIOptimizer',
        'hebo': 'src.optimizer_adaptors.hebo_opt.HeboOptimizer',
        'tidyhebo': 'src.optimizer_adaptors.tidyhebo_opt.TidyheboOptimizer',
    }
    _LOADED_CLASSES: Dict[str, Type['Model']] = {}

    _model_name: str = None  # Must be overridden by subclasses.

    def __init__(self, *, task_name: str, on_simplex: bool = False, task_minimization: bool = False):
        """
        Initialize the model with parameters shared by all adaptors.
        """
        self.task_name = task_name
        self.on_simplex = on_simplex
        self.task_minimization = task_minimization
        # *Note: If on simplex, simplex transformer will be intiialized later
        self.to_simplex_transformer: Optional[SimplexTransformer] = None
        self.observations: List[float] = []
        self._initialized = True

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not getattr(cls, '__abstractmethods__', None):
            if getattr(cls, "_model_name", None) is None:
                raise ValueError(f"{cls.__name__} must define class attribute '_model_name'.")
            name = cls._model_name
            if name not in cls._MODEL_REGISTRY:
                raise ValueError(
                    f"Model name '{name}' not found in _MODEL_REGISTRY. "
                    f"Available: {list(cls._MODEL_REGISTRY.keys())}"
                )

    @classmethod
    def _get_model_class(cls, model_type: str) -> Type['Model']:
        """
        Load model class (with all required imports from imported files)
        """
        if model_type in cls._LOADED_CLASSES:
            return cls._LOADED_CLASSES[model_type]

        import_path = cls._MODEL_REGISTRY.get(model_type)
        if import_path is None:
            raise ValueError(
                f"Unknown model type '{model_type}'. "
                f"Available: {list(cls._MODEL_REGISTRY.keys())}"
            )
        try:
            module_path, class_name = import_path.rsplit('.', 1)
            module = importlib.import_module(module_path)
            model_class = getattr(module, class_name)
            if not (isinstance(model_class, type) and hasattr(model_class, '_model_name')):
                raise TypeError(f"Class {class_name} is not a valid Model subclass")
            cls._LOADED_CLASSES[model_type] = model_class
            return model_class
        except (ImportError, AttributeError) as e:
            raise ImportError(
                f"Could not import model '{model_type}' from '{import_path}'. "
                f"Ensure the required dependencies are installed in the current environment. "
                f"Original error: {e}"
            ) from e

    @classmethod
    def from_config(cls, model_type: str, problem_cfg: Union[str, Path]) -> 'Model':
        """
        Creates model exemplar from task config file

        :param model_type: Model name (key in registry).
        :param problem_cfg: Path to YAML-file with task description
        :return: Model exemplar.
        """
        model_class = cls._get_model_class(model_type)

        with open(problem_cfg, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        params_list = config.get('params')
        if params_list is None:
            raise ValueError("Missing 'params' section in config.")

        on_simplex = config.get('on_simplex', False)
        task_minimization = config.get('less_is_better', False)
        task_name = config['task_name']

        if on_simplex:
            k = len(params_list)
            if k < 2:
                raise ValueError("Simplex requires at least 2 parameters.")
            optimizer_params = params_list[:-1]
            exemplar = model_class._from_config_impl(optimizer_params, task_name=task_name, on_simplex=on_simplex,
                                                    task_minimization=task_minimization)
            exemplar.to_simplex_transformer = SimplexTransformer(K=k)
            for p in params_list:
                if p.get('type') != 'continuous':
                    raise NotImplementedError(
                        "Simplex transformation supports only continuous parameters."
                    )
        else:
            exemplar = model_class._from_config_impl(params_list, task_name=task_name, on_simplex=on_simplex,
                                                    task_minimization=task_minimization)

        return exemplar

    @classmethod
    @abc.abstractmethod
    def _from_config_impl(cls, params: list, task_name: str, on_simplex: bool, task_minimization: bool) -> 'Model':
        """
        Creates model exemplar from task parameters.

        :param params: list[dict] with description of parameter from YAML file.
        :param on_simplex: flag is task defined on simplex param space.
        :param task_minimization: True, if task is a minimization task.
        :return: Model instance.
        """
        pass


    @abc.abstractmethod
    def _observe_impl(self, x0: XPoint, y0: YPoint) -> None:
        """
        Observes a one point x0 and y0, implying maximization task.
        *Note: All points are passed in optimizer params space, not in auxillary space, if on simplex.
        """
        pass

    def _observe_batch_impl(self, x_batch: XPoint, y_batch: YPoint) -> None:
        """
        Observes several points (x_batch, y_batch) in optimizer space.
        
        The method is optional to implement. By default raises NotImlpementedError if not overriden by subclasses
        """
        raise NotImplementedError("Batch observation not implemented for this optimizer.")
    def observe(self, x0: XPoint, y0: YPoint) -> None:
        """
        Observes point (of batch of points)
        """
        self._check_init()

        y0 = np.atleast_2d(np.asarray(y0))
        x0 = np.atleast_2d(np.asarray(x0))
        if y0.shape[0] != x0.shape[0]: raise ValueError(f"Sizes of x and y doesn't match. Given: x0.shape: {x0.shape}, y0.shape: {y0.shape}")

        # register observed values in observations here
        self.observations.extend(y0.ravel().tolist())

        if self.task_minimization:
            y0 = make_y_point(-y0)

        # x0 is on simples -> transform back
        if self.on_simplex:
            x0 = np.atleast_2d(x0)
            x0 = self.to_simplex_transformer.inverse_transform(x0)
            x0 = make_x_point(x0)  # (N_batches, -1)
        self._observe_impl(x0, y0) if x0.shape[0] == 1 else self._observe_batch_impl(x0, y0)


    @abc.abstractmethod
    def _suggest_impl(self) -> XPoint:
        """
        Return the next point in the original design space.
        """
        pass

    def _suggest_batch_impl(self, batch_size: int) -> XPoint:
        """
        Suggest array of points in the form (batch_size, -1) in optmiizer space.
        Must be transfered to simplex if required by task config.
        Raises NotImlpementedError if not overriden by subclasses.
        """
        raise NotImplementedError("Batch suggestion not implemented for this optimizer.")

    def _suggest_adaptive_batch_impl(self, adaptive_ceil: int) -> XPoint:
            """
            Suggest array of points in the form [min(adaptive_ceil, |PF|), -1](batch_size, -1) in optmiizer space.
            Must be transfered to simplex if required by task config.
            Raises NotImlpementedError if not overriden by subclasses.
            """
            raise NotImplementedError("Adaptive batch suggestion not implemented for this optimizer.")


    def suggest(self, batch_size: int = 1, adaptive_ceil: Optional[int] = None) -> XPoint:
        """
        Creates next point(s)
        Maps it to a simplex if requireed by task.
        If batch size > 1, requires 
        batch_size: int - if a fixed batch size should be returned
        adaptive_ceil: int | None - ceiling, like if the model wants to return adaptive batch size, but we don't want too many points; if set, the `batch_size` is ignored

        """
        self._check_init()

        # Dispatch according to the requested suggestion mode.
        if adaptive_ceil is not None:
            x_cand = self._suggest_adaptive_batch_impl(adaptive_ceil)
        else:
            x_cand = self._suggest_impl() if batch_size == 1 else self._suggest_batch_impl(batch_size)


        if self.on_simplex:
            if self.to_simplex_transformer is None:
                raise RuntimeError("Simplex transformer not initialized.")
            x_cand = self.to_simplex_transformer.transform(x_cand)

        return make_x_point(x_cand)


    @classmethod
    def available_models(cls) -> List[str]:
        return list(cls._MODEL_REGISTRY.keys())

    def get_history(self) -> np.ndarray:
        """
        Returns observed history (only 'y' values) without any implications
        Returns:
            np.ndarray of shape (1, -1)
        """
        self._check_init()
        return np.asarray(self.observations).reshape(1, -1)

    def _check_init(self):
        if not getattr(self, '_initialized', False):
            raise RuntimeError("Subclass must call super().__init__")
# endregion
