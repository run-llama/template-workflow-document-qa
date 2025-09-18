import functools
import os
import httpx

import dotenv
from llama_cloud.client import AsyncLlamaCloud

dotenv.load_dotenv()

# deployed agents may infer their name from the deployment name
# Note: Make sure that an agent deployment with this name actually exists
# otherwise calls to get or set data will fail. You may need to adjust the `or `
# name for development
agent_name = os.getenv("LLAMA_DEPLOY_DEPLOYMENT_NAME")
agent_name_or_default = agent_name or "test-proj"
# required for all llama cloud calls
api_key = os.environ["LLAMA_CLOUD_API_KEY"]
# get this in case running against a different environment than production
base_url = os.getenv("LLAMA_CLOUD_BASE_URL")
project_id = os.getenv("LLAMA_DEPLOY_PROJECT_ID")


def get_custom_client():
    return httpx.AsyncClient(
        timeout=60, headers={"Project-Id": project_id} if project_id else None
    )

@functools.lru_cache(maxsize=None)
def get_llama_cloud_client():
    return AsyncLlamaCloud(
        base_url=base_url,
        token=api_key,
        httpx_client=get_custom_client(),
    )
