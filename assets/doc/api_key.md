## OpenAI API Key Setup Guide
We use the OpenAI key to call `GPT-4o` to judge answer correctness. Please refer to the [official guide](https://platform.openai.com/api-keys) to setup. 

## Google API Key Setup Guide
We use the Google API key to call `google search tool`. Please refer to the [official guide](https://support.google.com/googleapi/answer/6158862?hl=en) to setup. 

## DashScope API Key Setup Guide

DashScope is Alibaba Cloud's large model service platform, providing API access to various large language models including Qwen (Tongyi Qianwen). In this project, we use the DashScope API key to call **Qwen-2.5-7B-Instruct** as the LLM engine for agents (except planner) and tools.



Please refer to the official guide: [https://help.aliyun.com/zh/model-studio/get-api-key](https://help.aliyun.com/zh/model-studio/get-api-key) (you may need to translate the page to English using your browser's translation feature).

> **Note**: Alternatively, you can serve Qwen-2.5-7B-Instruct model using vLLM or use other third-party providers like [together](https://api.together.xyz/).
