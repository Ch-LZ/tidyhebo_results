from dataclasses import dataclass


@dataclass(frozen=True)
class Task:
    name: str
    direction: str = "minimize"
    display_name: str = ""
    display_scale: float = 1.0
    display_sign: float = 1.0
    needle_levels: tuple[float, ...] = ()
    default_oracle_relpath: str | None = None

    def display(self, y):
        return self.display_sign * y / self.display_scale


TASKS = {
    "poisson_ratio": Task(
        name="poisson_ratio",
        display_name="Poisson ratio",
        needle_levels=(-1.2, -1.7),
        default_oracle_relpath="data/poisson/poisson_RF_trained.pkl",
    ),
    "thermoelectric": Task(
        name="thermoelectric",
        display_name="Thermoelectric figure of merit (ZT)",
        display_scale=50.0,
        display_sign=-1.0,
        needle_levels=(1.4,),
        default_oracle_relpath="data/thermoelectric/zt_RF_trained.pkl",
    ),
}
