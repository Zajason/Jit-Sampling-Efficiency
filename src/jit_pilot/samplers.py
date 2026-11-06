"""Keep official Euler/Heun step functions unchanged; add only an AB2 loop."""
from dataclasses import dataclass
import math
import torch

@dataclass(frozen=True)
class Configuration:
    name: str
    method: str
    steps: int
    schedule_power: float = 1.0

    def __post_init__(self):
        if self.method not in {'euler', 'heun', 'ab2'} or self.steps < 1:
            raise ValueError('Invalid solver or step count')
        if not math.isfinite(self.schedule_power) or self.schedule_power <= 0:
            raise ValueError('Schedule power must be finite and positive')

CONFIGS = tuple(Configuration(f"{method}_{steps}", method, steps)
                for method, steps in [("heun",50),("heun",25),("euler",50),
                                      ("euler",25),("ab2",50),("ab2",25)])

def expected_network_passes(method, steps):
    if method not in {"euler", "heun", "ab2"} or steps < 1:
        raise ValueError("Unknown method or invalid step count")
    return 2 * (2 * steps - 1 if method == "heun" else steps)

def ab2_update(z, velocity, previous, step, previous_step):
    """Variable-step AB2; first update is Euler."""
    if previous is None:
        return z + step * velocity
    ratio = step / previous_step
    return z + step * ((1 + ratio / 2) * velocity - ratio / 2 * previous)

@torch.no_grad()
def sample(model, noise, labels, config):
    # Same uniform grid, state dtype and endpoint convention as Denoiser.generate.
    z = noise.clone()
    times = torch.linspace(0., 1., config.steps + 1, device=z.device)
    # Explicit addition for schedule ablations. The uniform reference is untouched.
    if config.schedule_power != 1.0:
        times = times.pow(config.schedule_power)
    if not torch.all(times[1:] > times[:-1]):
        raise ValueError('Timesteps must be strictly increasing')
    times = times.view(-1, *([1] * z.ndim)).expand(-1, len(labels), -1, -1, -1)
    previous = previous_step = None
    for i in range(config.steps):
        t, next_t = times[i], times[i+1]
        if config.method == "heun" and i < config.steps - 1:
            z = model._heun_step(z, t, next_t, labels)
        elif config.method in {"heun", "euler"}:
            z = model._euler_step(z, t, next_t, labels)
        elif config.method == "ab2":
            step = next_t - t
            velocity = model._forward_sample(z, t, labels)
            z = ab2_update(z, velocity, previous, step, previous_step)
            previous, previous_step = velocity, step
        else:
            raise ValueError(config.method)
    return z
