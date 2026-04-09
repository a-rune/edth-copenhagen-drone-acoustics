from pathlib import Path
from typing import Literal

CLASS_TYPE = Literal["background", "drone", "helicopter"]
CLASSES: list[CLASS_TYPE] = ["background", "drone", "helicopter"]

RAW_DATA_DIR = Path(__file__).parent.parent.parent / "data" / "raw"
RAW_DATA_DIR.mkdir(exist_ok=True, parents=True)

EXAMPLES_DIR = Path(__file__).parent.parent.parent / "data" / "examples"

from hs_hackathon_drone_acoustics.localise import (  # noqa: E402
    MicArray,
    bandpass,
    classify_and_localise,
    coarse_to_fine_search,
    estimate_all_tdoas,
    gcc_phat,
    grid_search_tdoa_vectorised,
    localise_drone,
    preprocess_array,
    refine_with_nlls,
    sliding_tdoa,
    speed_of_sound,
    subpixel_peak,
    tdoa_error,
)

__all__ = [
    "CLASS_TYPE",
    "CLASSES",
    "EXAMPLES_DIR",
    "MicArray",
    "RAW_DATA_DIR",
    "bandpass",
    "classify_and_localise",
    "coarse_to_fine_search",
    "estimate_all_tdoas",
    "gcc_phat",
    "grid_search_tdoa_vectorised",
    "localise_drone",
    "preprocess_array",
    "refine_with_nlls",
    "sliding_tdoa",
    "speed_of_sound",
    "subpixel_peak",
    "tdoa_error",
]
