
def check_agent() -> bool:
    try:
        import litellm
        import prompt_toolkit
        import rich
        return True

    except ImportError as e:
        raise ImportError(f"failed to import agent dependencies, please try to install ghoshell-moss[agent]: {e}")
