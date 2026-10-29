_MEGATRON_TRAINING_AVAILABLE = None


def is_megatron_training_available():
    """
    Check if megatron.training module is available.

    Returns:
        bool: True if megatron.training is available, False otherwise.
    """
    global _MEGATRON_TRAINING_AVAILABLE
    if _MEGATRON_TRAINING_AVAILABLE is not None:
        return _MEGATRON_TRAINING_AVAILABLE

    try:
        import megatron.training
        _MEGATRON_TRAINING_AVAILABLE = True
    except ModuleNotFoundError:
        _MEGATRON_TRAINING_AVAILABLE = False

    return _MEGATRON_TRAINING_AVAILABLE
