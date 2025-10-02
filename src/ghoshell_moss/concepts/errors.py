class FatalError(Exception):
    pass


class InterpretError(Exception):
    pass


class CommandError(Exception):
    CANCEL_CODE = "canceled"
    NOT_AVAILABLE = "not_available"
    INVALID_USAGE = "invalid_usage"
    UNKNOWN_CODE = "unknown"

    def __init__(self, code: str, message: str = ""):
        self.code = code
        self.message = message
        super().__init__(f"Command error `{code}`: {message}")


class LoopStoppedError(Exception):
    """
    不需要记录的异常.
    """
    pass
