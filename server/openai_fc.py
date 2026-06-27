"""OpenAI function-calling 仿真：Qwen3 原生支持工具调用（Hermes 风格 <tool_call>），
但 oellm 的 llm 二进制只做普通对话、不套工具模板。这里：
- build_prompt：把 OpenAI messages + tools 拍平成「单行」prompt（含 Qwen3 工具说明）。
- parse_output：从模型输出里抽 <tool_call>{...}</tool_call> 还原成 OpenAI tool_calls。
"""

import json
import re
import uuid

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _flatten_content(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return " ".join(parts)
    return "" if content is None else str(content)


def build_prompt(messages: list[dict], tools: list[dict] | None = None) -> str:
    parts: list[str] = []
    if tools:
        tool_lines = " ".join(json.dumps(t, ensure_ascii=False) for t in tools)
        parts.append(
            "# Tools. You have these functions: <tools> " + tool_lines + " </tools>. "
            "If the user's request can be handled by any function, you MUST respond with ONLY "
            "the tool call(s) and nothing else, each EXACTLY as "
            '<tool_call>{"name": <function-name>, "arguments": <args-json-object>}</tool_call>. '
            "Only reply in plain text when no function applies."
        )
    for msg in messages:
        role = msg.get("role", "user")
        if role == "tool":
            name = msg.get("name") or msg.get("tool_call_id") or "tool"
            parts.append(f"[tool result of {name}]: {_flatten_content(msg.get('content'))}")
        elif role == "assistant" and msg.get("tool_calls"):
            calls = " ".join(
                f"<tool_call>{json.dumps({'name': c.get('function',{}).get('name'), 'arguments': c.get('function',{}).get('arguments')}, ensure_ascii=False)}</tool_call>"
                for c in msg["tool_calls"]
            )
            parts.append(f"Assistant: {calls}")
        else:
            parts.append(f"{role}: {_flatten_content(msg.get('content'))}")
    # /no_think：Qwen3 软开关，关闭思考链，直奔工具调用/答案，更快更稳
    if tools:
        parts.append("/no_think")
    parts.append("assistant:")
    return " ".join(p for p in parts if p).replace("\n", " ")


def parse_output(text: str) -> tuple[str, list[dict]]:
    cleaned = _THINK_RE.sub("", text)
    if "</think>" in cleaned:  # 思考块未配对（enable_thinking 仍开）时，丢弃 </think> 之前的思考残留
        cleaned = cleaned.rsplit("</think>", 1)[-1]
    tool_calls: list[dict] = []
    for match in _TOOL_CALL_RE.finditer(cleaned):
        try:
            obj = json.loads(match.group(1))
        except Exception:
            continue
        args = obj.get("arguments", {})
        if not isinstance(args, str):
            args = json.dumps(args, ensure_ascii=False)
        tool_calls.append({
            "id": "call_" + uuid.uuid4().hex[:20],
            "type": "function",
            "function": {"name": obj.get("name", ""), "arguments": args},
        })
    content = _TOOL_CALL_RE.sub("", cleaned).strip()
    return content, tool_calls
