# ABOUTME: Tracker adapter package for SORT, DeepSORT, and StrongSORT backends
# ABOUTME: Provides a unified TrackerAdapter interface across all three trackers

from .base import TrackerAdapter
from .sort_adapter import SortAdapter
from .deepsort_adapter import DeepSortAdapter
from .strongsort_adapter import StrongSortAdapter

TRACKER_REGISTRY: dict[str, type[TrackerAdapter]] = {
    "sort": SortAdapter,
    "deepsort": DeepSortAdapter,
    "strongsort": StrongSortAdapter,
}


def create_tracker(name: str, **kwargs) -> TrackerAdapter:
    """Create a tracker adapter by name."""
    if name not in TRACKER_REGISTRY:
        raise ValueError(f"Unknown tracker: {name}. Choose from {list(TRACKER_REGISTRY.keys())}")
    return TRACKER_REGISTRY[name](**kwargs)
