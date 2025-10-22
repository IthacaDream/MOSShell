from ghoshell_moss.channels.py_channel import PyChannel
import live2d.v3 as live2d
import asyncio

mouth_chan = PyChannel(name='mouth')

@mouth_chan.build.command()
async def open(size: float = 1.0):
    """
    open mouth

    @param size: 嘴巴打开大小，0 到 1 之间的浮点数
    """
    model = mouth_chan.client.container.force_fetch(live2d.LAppModel)

    model.SetParameterValue("ParamMouthOpenY", size)


@mouth_chan.build.command()
async def speek(duration: float = 5.0, speed: float = 0.02, max_open: float = 0.7, min_open: float = 0.0):
    """
    说话的嘴部动作

    @param duration: 动画持续时间，单位秒
    @param speed: 开合速度参数，值越大速度越快，默认0.02
    @param max_open: 最大张开程度，0 到 1 之间的浮点数，默认为0.7
    @param min_open: 最小张开程度，0 到 1 之间的浮点数，默认为0.0
    """
    import time
    
    # 参数验证
    if max_open > 1.0 or max_open < 0.0:
        max_open = 1.0
    if min_open < 0.0 or min_open > max_open:
        min_open = 0.2
    if speed <= 0:
        speed = 0.02
    
    # 动画帧间隔
    FRAME_INTERVAL = 0.01
    
    model = mouth_chan.client.container.force_fetch(live2d.LAppModel)
    
    # 保存当前嘴部状态
    original_value = 0.0  # 假设默认从闭合状态开始
    
    # 控制开合方向的状态变量
    delta = speed
    current_value = min_open
    
    start_time = time.time()
    time_up = False
    
    try:
        # 动画循环
        while True:
            current_time = time.time()
            
            # 检查是否已经到达设定时间
            if not time_up and current_time - start_time >= duration:
                time_up = True
            
            # 如果时间已到，并且嘴不是完全闭合状态，强制向闭合方向移动
            if time_up and current_value > 0:
                # 计算新位置，确保向0移动
                if current_value > 0:
                    new_value = current_value - speed
                    if new_value < 0:
                        new_value = 0
                else:
                    new_value = 0
            else:
                # 正常动画阶段，计算新位置
                new_value = current_value + delta
                
                # 检查是否到达端点，到达则改变方向
                if new_value <= min_open:
                    new_value = min_open
                    delta = -delta  # 改变方向向上
                elif new_value >= max_open:
                    new_value = max_open
                    delta = -delta  # 改变方向向下
            
            # 设置嘴部位置
            model.SetParameterValue("ParamMouthOpenY", new_value)
            current_value = new_value
            
            # 如果时间已到并且嘴已经完全闭合，结束动画
            if time_up and current_value <= 0:
                break
            
            # 等待一帧的时间
            await asyncio.sleep(FRAME_INTERVAL)
        
        # 确保最终状态是完全闭合
        model.SetParameterValue("ParamMouthOpenY", 0.0)
    except Exception as e:
        # 如果发生异常，恢复原始状态
        model.SetParameterValue("ParamMouthOpenY", original_value)
        raise e
