---
arguments: ''
description: 'PTT voice input — captures audio via PyAudio, runs Volcengine ASR, pushes recognized text as InputSignal to ghost mindflow via Zenoh session.'
executable: uv
respawn: false
script: main.py
workers: 1
---

PTT voice input app — captures audio, runs ASR, pushes recognized text as InputSignal to ghost mindflow