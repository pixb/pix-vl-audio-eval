# 音频-原稿拼音理解评估框架

评估**驱动本项目的大模型**对中文原稿的理解能力（多音字选音、文言读音等）。
音频读音以 Whisper 转写为参照；被评估模型只需给出它对原稿每个字的拼音理解，
框架自动逐字对比并生成报告。

## 原理

```
002.wav ──► Whisper ASR（统一固定）──► 识别文本 ──► 逐字拼音（音频实际读音）
                                                          │
原稿 ──► 被评估大模型理解 ──► understanding.json（逐字拼音）│
                                                          ▼
                                            逐字对比（rapidfuzz 对齐 + 多音字容错）
                                                          │
                                               report/report.md + report.json
```

- Whisper 侧：同音异形字（如识别成「徐普提」）按拼音等价，不算差异
- 评估的是**模型给出的拼音**与**音频读音**的偏差，不是 ASR 准确率

## 目录结构

```
pix-vl-audio-eval/
├── 002.wav                        # 测试音频
├── README.md
├── asr/                           # 统一 ASR（Whisper，固定不变）
│   ├── transcribe.py              #   音频 → 识别文本 + 逐字拼音
│   ├── requirements.txt
│   └── cache/                     #   转写缓存，运行后生成
└── models/
    └── _template/                 # 模型评估目录模板
        ├── TASK.md                #   给驱动 agent 的任务书
        ├── compare.py             #   对比脚本（LLM 理解 vs 音频读音）
        ├── understanding.example.json
        ├── requirements.txt
        └── report/                #   运行后生成
```

## 评估流程（每个模型一个目录，评估完删除换下一个）

```bash
# 0. 安装依赖（一次性）
pip install -r asr/requirements.txt -r models/_template/requirements.txt

# 1. 复制模板为被评估模型的目录（目录名 = 模型名）
cp -r models/_template models/mimo-v2.6-flash
# 编辑 models/mimo-v2.6-flash/TASK.md，把 <MODEL_NAME> 替换为实际模型名

# 2. 让驱动本项目的 agent 按 TASK.md 执行评估：
#    - 跑 asr/transcribe.py 生成音频拼音（有缓存则跳过）
#    - agent 阅读原稿，写出 understanding.json（它的拼音理解）
#    - 跑 compare.py 生成 report/

# 3. 评估者查看报告
cat models/mimo-v2.6-flash/report/report.md

# 4. 评估完毕，删除该模型目录，换下一个模型重复 1-3
rm -rf models/mimo-v2.6-flash
```

## 对比口径

- 逐字比较「模型理解拼音」vs「音频读音拼音」，无声调比对（`--strict-tone` 可开声调）
- 多音字按 pypinyin 词典全读音集合容错
- Whisper 漏识别的字不计入准确率，单独列出
- 退出码：`0` = 全部一致，`2` = 存在偏差

## 新增被评估模型

`cp -r models/_template models/<模型名>`，改 TASK.md 中的模型名即可。每个模型目录自包含，删除不影响其他。
