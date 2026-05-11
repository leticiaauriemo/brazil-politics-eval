MODELS = {
    "claude":      {"id": "anthropic/claude-sonnet-4-6",        "display": "Claude Sonnet 4.6"},
    "gpt4o":       {"id": "openai/gpt-4o",                      "display": "GPT-4o (paid)"},
    "gpt4omini":   {"id": "openai/gpt-4o-mini",                 "display": "ChatGPT Free (4o-mini)"},
    "grok":        {"id": "x-ai/grok-3",                        "display": "Grok 3"},
    "gemini":      {"id": "google/gemini-2.5-flash",             "display": "Gemini 2.5 Flash"},
    "deepseek":    {"id": "deepseek/deepseek-chat",              "display": "DeepSeek V3"},
    "llama":       {"id": "meta-llama/llama-3.3-70b-instruct",  "display": "Meta AI / Llama 3.3 70B"},
}

# Never self-score: 7-model cycle
# claude→gpt4o→gemini→deepseek→llama→grok→gpt4omini→claude
JUDGE_MAP = {
    "claude":    "gpt4o",
    "gpt4o":     "gemini",
    "gpt4omini": "claude",
    "grok":      "gpt4omini",
    "gemini":    "deepseek",
    "deepseek":  "llama",
    "llama":     "grok",
}
