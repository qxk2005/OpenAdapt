#!/usr/bin/env python3
"""测试 OpenAdapt 使用本地 OpenAI 兼容模型。

验证本地 LM Studio 模型与 OpenAdapt 的集成。
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# 加载 .env
env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)

print("=" * 60)
print("OpenAdapt 本地模型集成测试")
print("=" * 60)

# 1. 验证环境变量
print("\n1. 检查环境变量...")
print(f"   OPENAI_API_KEY: {'✅ 已设置' if os.getenv('OPENAI_API_KEY') else '❌ 未设置'}")
print(f"   OPENAI_BASE_URL: {os.getenv('OPENAI_BASE_URL', '❌ 未设置')}")

# 2. 测试 OpenAI 客户端连接
print("\n2. 测试本地模型连接...")
try:
    from openai import OpenAI
    client = OpenAI()
    print(f"   客户端 base_url: {client.base_url}")

    resp = client.chat.completions.create(
        model="google/gemma-4-26b-a4b-qat",
        messages=[{"role": "user", "content": "你好，请用中文回答：1+1等于几？"}],
        max_tokens=50,
        temperature=0.5,
    )
    print(f"   ✅ 模型响应: {resp.choices[0].message.content}")
    print(f"   模型: {resp.model}")
    print(f"   Token 使用: 输入={resp.usage.prompt_tokens}, 输出={resp.usage.completion_tokens}")
except Exception as e:
    print(f"   ❌ 连接失败: {e}")
    sys.exit(1)

# 3. 测试 OpenAdapt 配置
print("\n3. 检查 OpenAdapt 配置...")
try:
    from openadapt.config import settings
    print(f"   默认模型: {settings.default_model}")
    print(f"   OpenAI Key: {'✅ 已配置' if settings.openai_api_key else '⚠️  未在 OpenAdapt 配置中'}")
    print(f"   Capture 目录: {settings.capture_dir}")
except Exception as e:
    print(f"   ⚠️  配置加载异常: {e}")

# 4. 测试 OpenAdapt 各模块导入
print("\n4. 检查 OpenAdapt 模块...")
modules = {
    "openadapt": "核心 CLI",
    "openadapt_capture": "GUI 录制",
    "openadapt_ml": "ML 训练/推理",
    "openadapt_evals": "评估框架",
    "openadapt_viewer": "HTML 可视化",
    "openadapt_grounding": "UI 元素定位",
    "openadapt_retrieval": "演示检索",
}

for mod_name, desc in modules.items():
    try:
        __import__(mod_name)
        print(f"   ✅ {mod_name} ({desc})")
    except ImportError as e:
        print(f"   ❌ {mod_name} ({desc}): {e}")

# 5. 测试 ApiAgent 初始化（使用本地模型）
print("\n5. 测试 ApiAgent 使用本地模型...")
try:
    from openadapt_evals.agents.api_agent import ApiAgent

    agent = ApiAgent(
        provider="openai",
        model="google/gemma-4-26b-a4b-qat",
        temperature=0.5,
    )
    print(f"   ✅ ApiAgent 初始化成功")
    print(f"   Provider: {agent.provider}")
    print(f"   Model: {agent.model}")
    print(f"   客户端 base_url: {agent._client.base_url}")
except Exception as e:
    print(f"   ❌ ApiAgent 初始化失败: {e}")

# 6. 测试 Mock 评估（不需要 API 调用）
print("\n6. 测试 Mock 评估流程...")
try:
    from openadapt_evals import (
        SmartMockAgent,
        WAAMockAdapter,
        evaluate_agent_on_benchmark,
        compute_metrics,
    )

    agent = SmartMockAgent()
    adapter = WAAMockAdapter(num_tasks=3)

    results = evaluate_agent_on_benchmark(
        agent=agent,
        adapter=adapter,
        max_steps=5,
    )

    metrics = compute_metrics(results)
    print(f"   ✅ Mock 评估完成")
    print(f"   成功率: {metrics['success_rate']:.1%}")
    print(f"   平均步数: {metrics['avg_steps']:.1f}")
    print(f"   总任务数: {metrics['total_tasks']}")
except Exception as e:
    print(f"   ❌ Mock 评估失败: {e}")

print("\n" + "=" * 60)
print("测试完成！")
print("=" * 60)
print("""
后续使用:
  # 录制 GUI 操作
  openadapt capture start --name my-task

  # 使用本地模型评估
  openadapt eval run --agent api-openai --benchmark waa

  # 查看版本
  openadapt version
""")
