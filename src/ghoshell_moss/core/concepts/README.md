# Concepts

MOSS 架构最核心的抽象设计. 一切实现都基于抽象设计而来.

将这些抽象设计放到同一个目录, 方便: 

1. 人类理解所有的设计思想. 
2. AI 模型理解设计思想. 
3. 自迭代 AI 根据这些 interface, 能够更好地实现具体的功能.

需要理解 MOSS 架构设计思想, 建议先阅读这些设计文件. 建议的阅读顺序为: 

1. command
2. channel
3. shell
4. interpreter
5. errors
6. speech
7. states
8. topics