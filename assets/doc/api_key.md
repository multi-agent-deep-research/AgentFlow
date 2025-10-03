# API Keys Setup Guide

This guide provides detailed instructions on how to obtain API keys for all LLM providers and tools used in AgentFlow.

---

## 1. OpenAI API Key

**Purpose**: Access OpenAI's language models (GPT-4, GPT-3.5, etc.), used in AgentFlow for RAG summary and judging answer correctness.

**How to obtain**:
1. Visit [OpenAI Platform](https://platform.openai.com/)
2. Sign up/Log in to your account
3. Navigate to [API Keys page](https://platform.openai.com/api-keys)
4. Click "Create new secret key" to generate a new API key

**Available models**: [OpenAI Models Documentation](https://platform.openai.com/docs/models)

**Common model names**: `gpt-4o`,`gpt-4o-mini`, `gpt-4-turbo`, `gpt-3.5-turbo`

---

## 2. Google API Key (for Gemini LLM)

**Purpose**: Access Google's Gemini model series.

**How to obtain**:
1. Visit [Google AI Studio](https://aistudio.google.com/)
2. Log in with your Google account
3. Click "Get API key"
4. Create or select a Google Cloud project
5. Copy the generated API key

**Available models**: [Gemini Models Documentation](https://ai.google.dev/gemini-api/docs/models/gemini)

**Common model names**: `gemini-1.5-pro`, `gemini-2.5-flash`, `gemini-1.0-pro`

---

## 3. DashScope API Key (Alibaba Cloud) - Recommended for China & Singapore Region

**Purpose**: Access Alibaba Cloud's Qwen (Tongyi Qianwen) model series. In AgentFlow, we use DashScope to call **Qwen-2.5-7B-Instruct** as the LLM engine for agents (except planner) and tools.

**Recommended for**: Users in China region for better network connectivity and performance.

**How to obtain**:
1. Visit [Alibaba Cloud DashScope Console](https://dashscope.console.aliyun.com/)
2. Log in with your Alibaba Cloud account
3. Navigate to [API-KEY Management](https://dashscope.console.aliyun.com/apiKey)
4. Create a new API key

**Official guide**: [Get API Key](https://help.aliyun.com/zh/model-studio/get-api-key) (you may need to translate the page to English using your browser's translation feature)

**Available models**: [DashScope Model Documentation](https://help.aliyun.com/zh/dashscope/developer-reference/model-square)

**Common model names**: `qwen-turbo`, `qwen-plus`, `qwen-max`, `qwen2.5-7b-instruct`, `qwen2.5-72b-instruct`

> **Note**: For international users, we recommend using [Together AI](#4-together-api-key---recommended-for-international-users) to access Qwen-2.5-7B-Instruct model. Alternatively, you can serve the model locally using vLLM.

---

## 4. Together API Key - Recommended for More International Users

**Purpose**: Access open-source models on TogetherAI platform, including Qwen, Llama, Mixtral, etc.

**Recommended for**: International users who want to access Qwen-2.5-7B-Instruct and other open-source models with better global network connectivity.

**How to obtain**:
1. Visit [Together.ai](https://www.together.ai/)
2. Sign up/Log in to your account
3. Navigate to [Settings > API Keys](https://api.together.xyz/settings/api-keys)
4. Create a new API key

**Available models**: [Together Models Documentation](https://docs.together.ai/docs/inference-models)

**Common model names**:
- Qwen models: `Qwen/Qwen2.5-7B-Instruct`, `Qwen/Qwen2.5-72B-Instruct`
- Other models: `meta-llama/Llama-3-70b-chat-hf`, `mistralai/Mixtral-8x7B-Instruct-v0.1`

> **Important Note for Qwen Models**: Together AI offers both Turbo (quantized) and standard (non-quantized) versions of Qwen models. For best performance and accuracy, we recommend using the **non-quantized versions** (e.g., `Qwen/Qwen2.5-7B-Instruct` instead of `Qwen/Qwen2.5-7B-Instruct-Turbo`). The Turbo versions are faster but may have reduced quality due to quantization.

---

## 5. Anthropic API Key

**Purpose**: Access Anthropic's Claude model series.

**How to obtain**:
1. Visit [Anthropic Console](https://console.anthropic.com/)
2. Sign up/Log in to your account
3. Navigate to [API Keys page](https://console.anthropic.com/settings/keys)
4. Click "Create Key" to generate a new API key

**Available models**: [Anthropic Models Documentation](https://docs.anthropic.com/en/docs/about-claude/models)

**Common model names**: `claude-3-5-sonnet-20241022`, `claude-3-opus-20240229`, `claude-3-haiku-20240307`

---

## 6. DeepSeek API Key

**Purpose**: Access DeepSeek's language models.

**How to obtain**:
1. Visit [DeepSeek Platform](https://platform.deepseek.com/)
2. Sign up/Log in to your account
3. Navigate to [API Keys page](https://platform.deepseek.com/api_keys)
4. Create a new API key

**Available models**: [DeepSeek API Documentation](https://platform.deepseek.com/api-docs/)

**Common model names**: `deepseek-chat`, `deepseek-coder`

---

## 7. xAI API Key

**Purpose**: Access xAI's Grok models.

**How to obtain**:
1. Visit [xAI Console](https://console.x.ai/)
2. Sign up/Log in to your account
3. Navigate to API Keys page
4. Create a new API key

**Available models**: [xAI API Documentation](https://docs.x.ai/docs)

**Common model names**: `grok-beta`

---

## 8. Google API Key & CX (for Google Search Tool)

**Purpose**: Enable Google Programmable Search Engine for web searches in AgentFlow.

### Google API Key for Search

**How to obtain**:
1. Visit [Google Cloud Console](https://console.cloud.google.com/)
2. Create or select a project
3. Navigate to [APIs & Services > Credentials](https://console.cloud.google.com/apis/credentials)
4. Click "Create Credentials" > "API key"
5. Enable "Custom Search API": Visit [API Library](https://console.cloud.google.com/apis/library/customsearch.googleapis.com) and enable it

### Google CX (Search Engine ID)

**How to obtain**:
1. Visit [Programmable Search Engine](https://programmablesearchengine.google.com/)
2. Click "Add" to create a new search engine
3. Configure search scope (you can select "Search the entire web")
4. After creation, find the "Search engine ID" (CX) in the control panel

**Documentation**: [Custom Search JSON API Documentation](https://developers.google.com/custom-search/v1/overview)

---

## 9. Azure OpenAI Configuration (Optional)

**Purpose**: If using Azure-deployed OpenAI models instead of OpenAI directly.

### Required Configuration Parameters:

**AZURE_OPENAI_API_KEY**:
- Obtain from [Azure Portal](https://portal.azure.com/) > Azure OpenAI Resource > "Keys and Endpoint"

**AZURE_OPENAI_ENDPOINT**:
- Format: `https://<your-resource-name>.openai.azure.com/`
- Obtain from "Keys and Endpoint" page in Azure Portal

**AZURE_OPENAI_API_VERSION**:
- Common versions: `2024-02-15-preview`, `2023-12-01-preview`
- Reference: [Azure OpenAI API Version Documentation](https://learn.microsoft.com/en-us/azure/ai-services/openai/reference)

**AZURE_OPENAI_DEPLOYMENT**:
- Your deployment name created in Azure (create in "Deployments" page in Azure Portal)

**Documentation**: [Azure OpenAI Service Documentation](https://learn.microsoft.com/en-us/azure/ai-services/openai/)

---

## Important Notes

1. **Security**: All API keys are sensitive information - never expose them or commit to public repositories
2. **Costs**: Most API services are paid - understand pricing before use
3. **Quotas**: Some services have free tiers or rate limits - monitor your usage
4. **Environment Variables**: Copy `.env.template` to `.env` and fill in your actual API keys
5. **Regional Recommendations**:
   - **China users**: Use DashScope for Qwen models
   - **International users**: Use Together AI for Qwen models

---

## Quick Reference Table

| API Key | Purpose | Sign Up Link | Documentation |
|---------|---------|--------------|---------------|
| OPENAI_API_KEY | OpenAI GPT models | [platform.openai.com](https://platform.openai.com/) | [Docs](https://platform.openai.com/docs/models) |
| GOOGLE_API_KEY | Gemini models | [aistudio.google.com](https://aistudio.google.com/) | [Docs](https://ai.google.dev/gemini-api/docs/models/gemini) |
| DASHSCOPE_API_KEY | Qwen models (China) | [dashscope.console.aliyun.com](https://dashscope.console.aliyun.com/) | [Docs](https://help.aliyun.com/zh/dashscope/developer-reference/model-square) |
| TOGETHER_API_KEY | Qwen & open-source models (International) | [together.ai](https://www.together.ai/) | [Docs](https://docs.together.ai/docs/inference-models) |
| ANTHROPIC_API_KEY | Claude models | [console.anthropic.com](https://console.anthropic.com/) | [Docs](https://docs.anthropic.com/en/docs/about-claude/models) |
| DEEPSEEK_API_KEY | DeepSeek models | [platform.deepseek.com](https://platform.deepseek.com/) | [Docs](https://platform.deepseek.com/api-docs/) |
| XAI_API_KEY | Grok models | [console.x.ai](https://console.x.ai/) | [Docs](https://docs.x.ai/docs) |
| GOOGLE_API_KEY + CX | Google Search | [console.cloud.google.com](https://console.cloud.google.com/) | [Docs](https://developers.google.com/custom-search/v1/overview) |
