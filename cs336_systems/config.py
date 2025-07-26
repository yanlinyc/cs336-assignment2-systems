from dataclasses import dataclass

import tyro


@dataclass
class ModelConfig:
    vocab_size: int = 10_000
    rope_theta: float = 10000.0

    context_length: int = 256
    d_model: int = 512
    d_ff: int = 2048
    num_layers: int = 6
    num_heads: int = 8


PREDEFINED_CONFIGS = {
    "small": ModelConfig(d_model=768, d_ff=3072, num_layers=12, num_heads=12),
    "medium": ModelConfig(d_model=1024, d_ff=4096, num_layers=24, num_heads=16),
    "large": ModelConfig(d_model=1280, d_ff=5120, num_layers=36, num_heads=20),
    "xlarge": ModelConfig(d_model=1600, d_ff=6400, num_layers=48, num_heads=25),
    "2.7B": ModelConfig(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
}


@dataclass
class BenchmarkConfig:
    model_config_name: str = "small"
    model_config: ModelConfig | None = None
    batch_size: int = 4
    num_warmups: int = 5
    num_trials: int = 10
    device: str = "cuda"
    include_backward: bool = True
    time_unit: str = "seconds"  # Options: "seconds", "ms"
    debug: bool = False


def parse_arguments(default: BenchmarkConfig | None) -> BenchmarkConfig:
    config = tyro.cli(BenchmarkConfig, default=default)

    if config.model_config_name in PREDEFINED_CONFIGS:
        config.model_config = PREDEFINED_CONFIGS[config.model_config_name]

    assert config.model_config is not None, "Model configuration must be provided."

    print(f"Benchmark configuration: {config}")
    return config


def describe_time_summary(
    times: list[float], label: str, time_unit: str
) -> tuple[float, float, float, float]:
    """Calculate and display timing statistics."""
    # Convert times based on the specified unit
    unit_multiplier = 1000 if time_unit == "ms" else 1
    unit_label = "ms" if time_unit == "ms" else "seconds"

    converted_times = [t * unit_multiplier for t in times]

    mean_time = sum(converted_times) / len(converted_times)
    std_time = (sum((t - mean_time) ** 2 for t in converted_times) / len(converted_times)) ** 0.5
    min_time = min(converted_times)
    max_time = max(converted_times)

    print(f"{label} - Mean time ± Std: {mean_time:.6f} ± {std_time:.6f} {unit_label}")
    print(f"{label} - Min time: {min_time:.6f} {unit_label}, Max time: {max_time:.6f} {unit_label}")
    return mean_time, std_time, min_time, max_time
