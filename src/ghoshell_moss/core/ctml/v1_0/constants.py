from ghoshell_moss.core.concepts.channel import ChannelRuntime

__all__ = [
    'POSITION_ARGS_KEY', 'SCOPE_SHORTCUT', 'SCOPE_CHANNEL_NAME_KEY', 'CALL_ID_RESERVE_KEY', 'SCOPE_COMMAND_NAME',
    'CONTENT_COMMAND_NAME',
    'MAIN_CHANNEL_NAME', 'MAIN_CHANNEL_SHORTCUT',
    'MOSS_DYNAMIC', 'MOSS_STATIC',
    'SCOPE_ENTER_COMMAND_NAME',
    'SCOPE_EXIT_COMMAND_NAME',
]

MAIN_CHANNEL_NAME = '__main__'
MAIN_CHANNEL_SHORTCUT = ''
POSITION_ARGS_KEY = "_args"
SCOPE_SHORTCUT = '_'
SCOPE_COMMAND_NAME = '__scope__'
SCOPE_ENTER_COMMAND_NAME = ChannelRuntime.__scope_enter__.__name__
SCOPE_EXIT_COMMAND_NAME = ChannelRuntime.__scope_exit__.__name__
CONTENT_COMMAND_NAME = ChannelRuntime.__content__.__name__
CALL_ID_RESERVE_KEY = '_cid'
SCOPE_CHANNEL_NAME_KEY = 'channel'

MOSS_DYNAMIC = 'moss_dynamic'
MOSS_STATIC = 'moss_static'
