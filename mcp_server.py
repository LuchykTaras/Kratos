import os
from typing import Any
import httpx
from mcp.server.mcpserver import MCPServer

mcp = MCPServer("KRATOS-Neural-Architect-SPPR")

API_BASE_URL = os.getenv("KRATOS_API_URL", "http://127.0.0.1:8000").rstrip("/")


@mcp.tool()
async def evaluate_neural_architecture(
    tz_text: str,
    domain: str = "Computer Vision (Edge / Raspberry Pi)",
    benchmark_name: str = "MobileNetV3-Small",
    x1_complexity: float = 25.0,
    x2_latency: float = 20.0,
    x3_memory: float = 30.0,
    x5_robustness: float = 85.0,
    x7_generalization: float = 80.0,
    x10_multimodal: float = 30.0,
) -> dict[str, Any]:
    """
    Виконує нечітке логічне виведення (Частина 1): оцінює ТЗ, обчислює
    інтегральну Sing-метрику [-1.0; +1.0], перевіряє поріг евристики (50%)
    та повертає рекомендовану архітектуру нейромережі зі специфікацією шарів.
    """
    payload = {
        "tz_text": tz_text,
        "domain": domain,
        "benchmark_name": benchmark_name,
        "x1_complexity": x1_complexity,
        "x2_latency": x2_latency,
        "x3_memory": x3_memory,
        "x5_robustness": x5_robustness,
        "x7_generalization": x7_generalization,
        "x10_multimodal": x10_multimodal,
    }
    async with httpx.AsyncClient(timeout=15.0) as client:
        response = await client.post(f"{API_BASE_URL}/api/evaluate", json=payload)
        response.raise_for_status()
        return response.json()


@mcp.tool()
async def get_working_memory_facts() -> dict[str, Any]:
    """Повертає всі активні факти з робочої пам'яті експертної системи (Частина 2)."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(f"{API_BASE_URL}/api/facts")
        response.raise_for_status()
        return response.json()


@mcp.tool()
async def assert_fact(template: str, slots: dict[str, Any]) -> dict[str, Any]:
    """Додає новий факт (операція CLIPS assert): 'neural_architecture', 'tz_requirement' або 'hardware_constraint'."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.post(
            f"{API_BASE_URL}/api/facts/assert",
            json={"template": template, "slots": slots},
        )
        response.raise_for_status()
        return response.json()


@mcp.tool()
async def modify_fact(num_id: int, slot_name: str, new_value: Any) -> dict[str, Any]:
    """Змінює значення слоту в існуючому факті за індексом num_id (операція modify)."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.post(
            f"{API_BASE_URL}/api/facts/modify",
            json={"num_id": num_id, "slot_name": slot_name, "new_value": new_value},
        )
        response.raise_for_status()
        return response.json()


@mcp.tool()
async def retract_facts(num_ids: list[int]) -> dict[str, Any]:
    """Вилучає факти з робочої пам'яті за списком індексів num_ids (операція retract)."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.post(
            f"{API_BASE_URL}/api/facts/retract",
            json={"num_ids": num_ids},
        )
        response.raise_for_status()
        return response.json()


@mcp.tool()
async def reset_working_memory() -> dict[str, Any]:
    """Скидає робочу пам'ять до початкового набору фактів deffacts f-1..f-6 (операція reset)."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.post(f"{API_BASE_URL}/api/facts/reset")
        response.raise_for_status()
        return response.json()


if __name__ == "__main__":
    mcp.run(transport="stdio")