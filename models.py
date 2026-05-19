# Models grouped by family for display purposes
MODELS = {
    # Anthropic — all three tiers
    "claude_haiku":   {"id": "anthropic/claude-haiku-4.5",              "display": "Claude Haiku 4.5",        "family": "Anthropic"},
    "claude_sonnet":  {"id": "anthropic/claude-sonnet-4.6",             "display": "Claude Sonnet 4.6",       "family": "Anthropic"},
    "claude_opus":    {"id": "anthropic/claude-opus-4.7",               "display": "Claude Opus 4.7",         "family": "Anthropic"},

    # OpenAI — free tier, paid tier, newer
    "gpt4omini":      {"id": "openai/gpt-4o-mini",                      "display": "ChatGPT Free (4o-mini)",  "family": "OpenAI"},
    "gpt4o":          {"id": "openai/gpt-4o",                           "display": "GPT-4o",                  "family": "OpenAI"},
    "gpt41":          {"id": "openai/gpt-4.1",                          "display": "GPT-4.1",                 "family": "OpenAI"},
    "gpt5":           {"id": "openai/gpt-5",                            "display": "GPT-5",                   "family": "OpenAI"},

    # Google
    "gemini_flash":   {"id": "google/gemini-2.5-flash",                 "display": "Gemini 2.5 Flash",        "family": "Google"},
    "gemini_pro":     {"id": "google/gemini-2.5-pro",                   "display": "Gemini 2.5 Pro",          "family": "Google"},

    # xAI
    "grok4":          {"id": "x-ai/grok-4.3",                           "display": "Grok 4.3",                "family": "xAI"},

    # Meta — Llama (Meta AI / WhatsApp baseline)
    "llama33":        {"id": "meta-llama/llama-3.3-70b-instruct",       "display": "Meta AI / Llama 3.3 70B", "family": "Meta"},
    "llama4":         {"id": "meta-llama/llama-4-maverick",             "display": "Llama 4 Maverick",        "family": "Meta"},

    # DeepSeek
    "deepseek":       {"id": "deepseek/deepseek-chat",                  "display": "DeepSeek V3",             "family": "DeepSeek"},
    "deepseek_r1":    {"id": "deepseek/deepseek-r1",                    "display": "DeepSeek R1 (reasoning)", "family": "DeepSeek"},

    # Perplexity — only model with live web search
    "perplexity":     {"id": "perplexity/sonar",                        "display": "Perplexity Sonar",        "family": "Perplexity"},

    # Qwen (Alibaba) — largest Chinese model, interesting for PT training data
    "qwen":           {"id": "qwen/qwen3-235b-a22b",                    "display": "Qwen3 235B",              "family": "Qwen"},

    # Mistral — European model
    "mistral":        {"id": "mistralai/mistral-large-2411",            "display": "Mistral Large",           "family": "Mistral"},
}

# Single judge: gemini_flash for all (except itself, which uses gpt4omini)
JUDGE_MAP = {
    "claude_haiku":  "gemini_flash",
    "claude_sonnet": "gemini_flash",
    "claude_opus":   "gemini_flash",
    "gpt4omini":     "gemini_flash",
    "gpt4o":         "gemini_flash",
    "gpt41":         "gemini_flash",
    "gpt5":          "gemini_flash",
    "gemini_flash":  "gpt4omini",
    "gemini_pro":    "gemini_flash",
    "grok4":         "gemini_flash",
    "llama33":       "gemini_flash",
    "llama4":        "gemini_flash",
    "deepseek":      "gemini_flash",
    "deepseek_r1":   "gemini_flash",
    "perplexity":    "gemini_flash",
    "qwen":          "gemini_flash",
    "mistral":       "gemini_flash",
}

# Core models (most used by Brazilians — run these first)
CORE_MODELS = ["gpt4omini", "gpt4o", "gemini_flash", "llama33", "claude_sonnet"]

# Extended models (run after core)
EXTENDED_MODELS = [k for k in MODELS if k not in CORE_MODELS]
