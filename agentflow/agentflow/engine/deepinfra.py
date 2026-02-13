try:
    from openai import OpenAI
except ImportError:
    raise ImportError("Please install the openai package by running `pip install openai`, and add 'DEEPINFRA_API_KEY' to your environment variables.")

import os
import platformdirs
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from typing import List, Union
from .base import EngineLM, CachedEngine


class ChatDeepInfra(EngineLM, CachedEngine):
    DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."

    def __init__(
        self,
        model_string,
        use_cache: bool=False,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        is_multimodal: bool=False,
        ):

        self.model_string = model_string
        self.use_cache = use_cache
        self.system_prompt = system_prompt
        self.is_multimodal = is_multimodal

        if self.use_cache:
            root = platformdirs.user_cache_dir("agentflow")
            cache_path = os.path.join(root, f"cache_deepinfra_{model_string}.db")
            super().__init__(cache_path=cache_path)

        if os.getenv("DEEPINFRA_API_KEY") is None:
            raise ValueError("Please set the DEEPINFRA_API_KEY environment variable.")
        self.client = OpenAI(
            api_key=os.getenv("DEEPINFRA_API_KEY"),
            base_url="https://api.deepinfra.com/v1/openai"
        )

    @retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(5))
    def generate(self, content: Union[str, List[Union[str, bytes]]], system_prompt=None, **kwargs):
        if isinstance(content, list) and len(content) == 1:
            content = content[0]
        if isinstance(content, str):
            return self._generate_text(content, system_prompt=system_prompt, **kwargs)

    def _generate_text(
        self, prompt, system_prompt=None, temperature=0, max_tokens=4000, top_p=0.99, response_format=None
    ):
        sys_prompt_arg = system_prompt if system_prompt else self.system_prompt

        if self.use_cache:
            cache_key = sys_prompt_arg + prompt
            cache_or_none = self._check_cache(cache_key)
            if cache_or_none is not None:
                return cache_or_none
        
        response = self.client.chat.completions.create(
            model=self.model_string,
            messages=[
                {"role": "system", "content": sys_prompt_arg},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p
        )
        response = response.choices[0].message.content

        if self.use_cache:
            self._save_cache(cache_key, response)
        return response

    def __call__(self, prompt, **kwargs):
        return self.generate(prompt, **kwargs)
    