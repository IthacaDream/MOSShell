from ghoshell_moss.channels.py_channel import PyChannel
import time
import asyncio


left_arm_chan = PyChannel('left_arm')
right_arm_chan = PyChannel('right_arm')


@left_arm_chan.build.command()
async def raise_left_arm(duration: float = 1.5):
    """抬手"""
    start_time = time.time()
    while time.time() - start_time < duration:
        progress = (time.time() - start_time) / duration
        # 根据模型文件，使用PARAM_ARM_L_01参数来控制左上臂
        # 角度从0度到90度，实现抬手动作
        angle = 90 * progress
        model.SetParameterValue('PARAM_ARM_L_01', angle)
        await asyncio.sleep(0.016)


@right_arm_chan.build.command()
async def raise_right_arm(duration: float = 1.5):
    """抬手"""
    start_time = time.time()
    while time.time() - start_time < duration:
        progress = (time.time() - start_time) / duration
        # 根据模型文件，使用PARAM_ARM_R_01参数来控制右上臂
        # 角度从0度到90度，实现抬手动作
        angle = 90 * progress
        model.SetParameterValue('PARAM_ARM_R_01', angle)
        await asyncio.sleep(0.016)