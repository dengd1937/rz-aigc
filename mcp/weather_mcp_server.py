import os
import httpx
from dotenv import load_dotenv
from fastmcp import FastMCP

load_dotenv()

HEFENG_API_HOST = "https://kn759gf7ub.re.qweatherapi.com"

mcp = FastMCP("weather-server")


async def _get_location_id(location: str) -> str:
    async with httpx.AsyncClient() as http_client:
        response = await http_client.get(
            f"{HEFENG_API_HOST}/geo/v2/city/lookup",
            params={
                "location": location,
                "key": os.getenv("HEFENG_API_KEY")
            }
        )
        response.raise_for_status()
        data = response.json()
        return data["location"][0]["id"]


@mcp.tool()
async def get_weather(location: str, days: str = "3d") -> dict:
    """
    获取指定位置的天气信息
    
    :param location: 位置名称，例如 "北京"
    :param days: 预报天数，支持最多30天预报，可选值：3d 3天预报、7d 7天预报、10d 10天预报、15d 15天预报、30d 30天预报。
    :return: 包含天气信息的字典
    """
    location_id = await _get_location_id(location)
    async with httpx.AsyncClient() as http_client:
        response = await http_client.get(
            f"{HEFENG_API_HOST}/v7/weather/{days}",
            params={
                "location": location_id,
                "key": os.getenv("HEFENG_API_KEY")
            }
        )
        response.raise_for_status()
        data = response.json()
        return data


if __name__ == "__main__":
    mcp.run(transport="sse")
