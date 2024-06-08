import asyncio
import logging
from asyncio import Semaphore
import random
import httpx
from tqdm import tqdm as tqdm_sync
import utils
from progress_manager import ProgressManager

from prompt_constructor import read_prompts, construct_prompts

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S', )
logger = logging.getLogger()

API_TOKEN = "NXm9ucb4t5WJsr5oH0D5UdleIFZl6hS321UfojoJnH0BnxFxxAmnmOpb6J69KZSY"
JSON_HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_TOKEN}",
}

random.seed(42)


def get_seed():
    return random.randint(0, 1125899906842623)


def get_gen_payload(**kwargs):
    payload = {
        "width": 1024,
        "height": 1024,
        "prompt": "",
        "negative_prompt": "weird, deformed, bad, cartoon, blurry",
        "seed": 42,
        "steps": 5,
        "cfg_scale": 1.7,
        "restore_faces": False,
        "nsfw_skip": True,
        "return_img": True,
    }
    return payload | kwargs


async def fetch(url: str, key: int, payload: dict, metadata, semaphore: Semaphore,
                pm: ProgressManager, pbar: tqdm_sync):
    async with semaphore:
        print(f"Fetching image number {key}")
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload, headers=JSON_HEADERS, timeout=70)
            res = response.json()
            img = res['images'][0]
            img = utils.base64_2_img(img)
            pm.update_progress(key, payload, metadata, img)
            del img
            del res
            pbar.update(1)


async def run_requests(url: str, constructed_prompts: list[tuple], max_concurrent_requests: int, pm: ProgressManager):
    semaphore = Semaphore(max_concurrent_requests)
    total_tasks = len(constructed_prompts)
    tasks = []
    with tqdm_sync(total=total_tasks, desc="Creating images") as pbar:
        for i, (prompt, metadata) in enumerate(constructed_prompts):
            seed = get_seed()
            if not pm.key_exists(i):
                payload = get_gen_payload(prompt=prompt, seed=seed)
                task = asyncio.create_task(fetch(url, i, payload, metadata, semaphore, pm, pbar))
                tasks.append(task)
            else:
                pbar.update(1)
            await asyncio.sleep(0.001)
        await asyncio.gather(*tasks)


def main():
    raw_prompts = read_prompts('data/prompts_raw.txt')
    constructed_prompts = construct_prompts(raw_prompts)[:6000]
    url = "http://34.72.181.10:3000/generate"
    max_concurrent_requests = 12
    logger.info(f"Number of constructed prompts: {len(constructed_prompts)}")
    logger.info(f"Max concurrent requests: {max_concurrent_requests}")
    pm = ProgressManager("data/progress.json", "data/images")
    asyncio.run(run_requests(url, constructed_prompts, max_concurrent_requests, pm))


if __name__ == '__main__':
    main()
