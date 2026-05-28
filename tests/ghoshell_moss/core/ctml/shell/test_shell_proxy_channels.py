import pytest
from ghoshell_moss.core.ctml.shell import new_ctml_shell
from ghoshell_moss.core.duplex.thread_channel import create_thread_bridge
from ghoshell_moss.core import PyChannel
import asyncio


@pytest.mark.asyncio
async def test_shell_with_virtual_sub_depth_channel():
    provider_main = PyChannel(name="provider")
    static_sub = PyChannel(name="static_sub")
    virtual_sub = PyChannel(name="virtual_sub")

    @virtual_sub.build.command()
    async def foo():
        return 123

    virtual_sub_depth_2 = PyChannel(name="virtual_sub_depth_2")

    provider_main.import_channels(static_sub)

    provider, proxy = create_thread_bridge('proxy')
    shell = new_ctml_shell("test")
    shell.main_channel.import_channels(proxy)

    async with provider.arun(provider_main):
        async with shell:
            await shell.wait_connected("proxy")
            assert len(shell.channel_metas()) == 3
            # 添加动态
            provider_main.add_virtual_channel(virtual_sub)
            await shell.refresh_metas()
            # 拿到了新的节点.
            assert len(shell.channel_metas()) == 4
            virtual_sub.add_virtual_channel(virtual_sub_depth_2)
            # 继续添加动态节点.
            await shell.refresh_metas()
            metas = shell.channel_metas()
            assert len(metas) == 5
            assert len(shell.channel_metas()) == 5
            commands = shell.commands()
            assert 'proxy.virtual_sub' in commands
            assert 'foo' in commands['proxy.virtual_sub']
            count = 0
            command = await shell.get_command("proxy.virtual_sub", "foo")
            assert command is not None
            assert command.meta().available

            # 判断 provider 和 proxy 都有正确的命令.
            proxy_count = 0
            for path, meta in shell.channel_metas().items():
                if path.startswith("proxy"):
                    proxy_count += 1
                    assert meta.proxy
                assert meta.available
                for command in meta.commands:
                    assert command.available
                count += 1
            assert count == 5
            assert shell.runtime.self_meta().proxy is False
            assert proxy_count == 4
            assert 'virtual_sub' in provider.runtime.commands()
            assert 'foo' in provider.runtime.commands()['virtual_sub']
            cmd = provider.runtime.get_command("virtual_sub:foo")
            assert cmd is not None

            # 少一个节点.
            virtual_sub.remove_virtual_channel(virtual_sub_depth_2.name())
            await shell.refresh_metas()
            assert len(shell.channel_metas()) == 4

            async with shell.interpreter_in_ctx() as i:
                i.feed("<proxy.virtual_sub:foo />")
                i.commit()
                await i.wait_compiled()
                tasks = await i.wait_tasks()
                assert len(tasks) == 1
                t = list(tasks.values())[0]
                e = t.exception()
                assert e is None
                assert t.success()
                assert t.result() == 123

            command_group = shell.commands()
            assert 'proxy.virtual_sub' in command_group
            assert 'foo' in command_group['proxy.virtual_sub']


@pytest.mark.asyncio
async def test_shell_proxy_delta_calls_in_double_proxy():
    provider_1_main = PyChannel(name="provider1")
    provider_1, proxy_1 = create_thread_bridge('proxy_1')
    provider_2_main = PyChannel(name="provider2")

    provider_2_main.import_channels(proxy_1)
    provider_2, proxy_2 = create_thread_bridge('proxy_2')

    shell = new_ctml_shell(
        "test",
    )
    shell.main_channel.import_channels(proxy_2)

    got = ''

    @provider_1_main.build.command()
    async def chunks(chunks__):
        nonlocal got
        async for c in chunks__:
            got += c

    # provider 1 将 provider 1 main 提供出来,
    async with provider_1.arun(provider_1_main):
        # 启动 provider 2 main 时启动了 proxy.
        async with provider_2.arun(provider_2_main):
            async with shell:
                await shell.wait_connected("proxy_2")
                async with shell.interpreter_in_ctx() as i:
                    i.feed("<proxy_2.proxy_1:chunks>hello world</proxy_2.proxy_1:chunks>")
                    i.commit()
                    await i.wait_stopped()
                    i.raise_exception()
    assert got == 'hello world'


@pytest.mark.asyncio
async def test_shell_proxy_channel_with_other_magic_command():
    provider_main = PyChannel(name="provider")
    provider, proxy = create_thread_bridge('proxy')

    got = ''

    @provider_main.build.command()
    async def __hello__():
        nonlocal got
        got = "world"
        return got

    shell = new_ctml_shell()
    shell.main_channel.import_channels(proxy)

    async with provider.arun(provider_main):
        async with shell:
            await shell.wait_connected("proxy")
            # 魔术方法不对外暴露即可.
            for msg in shell.dynamic_messages():
                assert "__hello__" not in msg.to_content_string()

            assert "__hello__" not in shell.static_messages()
            assert "__hello__" not in shell.meta_instruction()


@pytest.mark.asyncio
async def test_shell_proxy_channel_with_magic_delta_calls():
    provider_main = PyChannel(name="provider")
    provider, proxy = create_thread_bridge('proxy')

    got = ''

    @provider_main.build.command()
    async def __magic__(chunks__):
        nonlocal got
        async for c in chunks__:
            got += c

    shell = new_ctml_shell(
        "test",
    )
    shell.main_channel.import_channels(proxy)

    async with provider.arun(provider_main):
        async with shell:
            await shell.wait_connected("proxy")
            assert "proxy" in shell.commands()
            async with shell.interpreter_in_ctx() as i:
                i.feed("<proxy:__magic__>hello world</proxy:__magic__>")
                i.commit()
                await i.wait_tasks()
    assert got == 'hello world'


@pytest.mark.asyncio
async def test_shell_proxy_channel_with_delta_calls():
    provider_main = PyChannel(name="provider")
    provider, proxy = create_thread_bridge('proxy')

    got = ''

    @provider_main.build.command()
    async def chunks(chunks__):
        nonlocal got
        async for c in chunks__:
            got += c

    shell = new_ctml_shell(
        "test",
    )
    shell.main_channel.import_channels(proxy)

    errors = []

    def report(err):
        errors.append(err)

    provider.on_error(report)

    async with provider.arun(provider_main):
        async with shell:
            await shell.wait_connected("proxy")
            async with shell.interpreter_in_ctx() as i:
                i.feed("<proxy:chunks>hello world</proxy:chunks>")
                i.commit()
                try:
                    await i.wait_compiled()
                except Exception as e:
                    print(e)
                tasks = i.compiled_tasks()
                for task in tasks.values():
                    assert task.meta.available
                assert len(tasks) == 1
                await shell.clear()
                interpretation = await i.wait_stopped()
                assert len(interpretation.failed_tasks) == 0
    assert len(errors) == 0


@pytest.mark.asyncio
async def test_shell_proxy_channel_with_remote_content_command():
    provider_main = PyChannel(name="provider")
    provider, proxy = create_thread_bridge('proxy')

    got = ''

    @provider_main.build.content_command
    async def chunks(chunks__):
        nonlocal got
        async for c in chunks__:
            got += c

    shell = new_ctml_shell()
    shell.main_channel.import_channels(proxy)
    async with provider.arun(provider_main):
        async with shell:
            await shell.wait_connected("proxy")
            await shell.refresh_metas()
            metas = shell.channel_metas()
            assert 'proxy' in metas
            async with shell.interpreter_in_ctx() as i:
                i.feed("<proxy:__content__>hello world</proxy:__content__>")
                i.commit()
                await i.wait_tasks()
    assert got == 'hello world'


@pytest.mark.asyncio
async def test_shell_proxy_channel_with_content_command_that_not_exists():
    provider_main = PyChannel(name="provider")
    provider, proxy = create_thread_bridge('proxy')

    shell = new_ctml_shell()
    shell.main_channel.import_channels(proxy)

    errors = []

    def print_error(err):
        errors.append(err)

    provider.on_error(print_error)
    async with provider.arun(provider_main):
        async with shell:
            await shell.wait_connected("proxy")
            await shell.refresh_metas()
            metas = shell.channel_metas()
            assert 'proxy' in list(metas.keys())
            async with shell.interpreter_in_ctx() as i:
                i.feed("<proxy:__content__>hello world</proxy:__content__>")
                i.commit()
                tasks = await i.wait_tasks(timeout=2, throw_task_error=True)
                for task in tasks.values():
                    assert task.exception() is None, "++++++ the task is:" + task.caller_name()
                i.raise_exception()

    assert len(errors) == 0


@pytest.mark.asyncio
async def test_shell_proxy_channel_with_content_command_by_scope():
    provider_main = PyChannel(name="provider")
    provider, proxy = create_thread_bridge('proxy')

    shell = new_ctml_shell()
    shell.main_channel.import_channels(proxy)

    errors = []

    got = ''

    @provider_main.build.content_command
    async def chunks(chunks__):
        try:
            nonlocal got
            async for c in chunks__:
                got += c
        except asyncio.CancelledError:
            got = 'canceled'

    def print_error(err):
        errors.append(err)

    provider.on_error(print_error)
    async with provider.arun(provider_main):
        async with shell:
            await shell.wait_connected("proxy")
            await shell.refresh_metas()
            metas = shell.channel_metas()
            assert 'proxy' in metas
            async with shell.interpreter_in_ctx() as i:
                i.feed("<_ channel='proxy'>hello world</_>")
                i.commit()
                tasks = await i.wait_tasks(timeout=10)
                assert len(tasks) == 3
                i.raise_exception()

    assert len(errors) == 0
    assert got == 'hello world'


@pytest.mark.asyncio
async def test_shell_proxy_channel_with_scope_call():
    provider_main = PyChannel(name="provider")
    provider, proxy = create_thread_bridge('proxy')

    got = []

    @provider_main.build.command()
    async def foo():
        got.append(1)
        await asyncio.sleep(0.1)
        return "hello"

    shell = new_ctml_shell()
    shell.main_channel.import_channels(proxy)

    async with provider.arun(provider_main):
        async with shell:
            await shell.wait_connected("proxy")
            assert provider.runtime.is_running()
            assert shell.runtime.is_running()
            proxy_runtime = shell.runtime.fetch_sub_runtime('proxy')
            assert proxy_runtime.is_running()

            assert len(got) == 0
            async with shell.interpreter_in_ctx() as i:
                i.feed("<_ channel='proxy' until='any'><foo /><foo /><foo /></_>")
                i.commit()
                tasks = await i.wait_tasks(timeout=1)
                i.raise_exception()
            assert len(got) == 1
            await shell.clear()

            got.clear()
            async with shell.interpreter_in_ctx() as i:
                i.feed("<_ channel='proxy' until='all'><foo /><foo /><foo /></_>")
                i.commit()
                tasks = await i.wait_tasks(timeout=1)
                i.raise_exception()
            assert len(got) == 3
            await shell.clear()

            got.clear()
            async with shell.interpreter_in_ctx() as i:
                i.feed("<_ channel='proxy' until='all'> <_><foo /></_> <foo />  <_><foo /></_>  </_> <proxy:foo />")
                i.commit()
                tasks = await i.wait_tasks(timeout=1)
                for task in tasks.values():
                    if exp := task.exception():
                        print(repr(exp))
                i.raise_exception()
            assert len(got) == 4
