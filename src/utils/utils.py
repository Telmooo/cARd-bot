from typing import Any, Dict, List, Tuple


def flatten_dictionary(
    dictionary: Dict[str, Any],
    parent_key: str = "",
    separator: str = "."
) -> Dict[str, Any]:
    items: List[Tuple[str, Any]] = []
    for key, value in dictionary.items():
        flattened_key = parent_key + separator + key if parent_key else key
        if isinstance(value, dict):
            items.extend(flatten_dictionary(value, parent_key=flattened_key, separator=separator).items())
        else:
            items.append((flattened_key, value))

    return dict(items)
