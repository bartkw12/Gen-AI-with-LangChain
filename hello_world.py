from __future__ import annotations

import os
import sys

from langchain_openai import AzureChatOpenAI
from pydantic import SecretStr


DEFAULT_PROMPT = "Tell me a joke about trains."
REQUIRED_ENV_VARS = (
	"AZURE_OPENAI_ENDPOINT",
	"AZURE_OPENAI_API_KEY",
	"AZURE_OPENAI_API_VERSION",
	"AZURE_OPENAI_DEPLOYMENT",
)


def get_required_env(name: str) -> str:
	value = os.getenv(name, "").strip()
	if value:
		return value
	raise RuntimeError(
		f"Missing environment variable: {name}. "
		"Set your Azure OpenAI endpoint, API key, API version, and deployment before running this script."
	)


def create_llm() -> AzureChatOpenAI:
	missing = [name for name in REQUIRED_ENV_VARS if not os.getenv(name, "").strip()]
	if missing:
		raise RuntimeError(f"Missing environment variables: {', '.join(missing)}")

	return AzureChatOpenAI(
		azure_endpoint=get_required_env("AZURE_OPENAI_ENDPOINT"),
		api_key=SecretStr(get_required_env("AZURE_OPENAI_API_KEY")),
		api_version=get_required_env("AZURE_OPENAI_API_VERSION"),
		azure_deployment=get_required_env("AZURE_OPENAI_DEPLOYMENT"),
	)


def main() -> None:
	prompt = " ".join(sys.argv[1:]).strip() or DEFAULT_PROMPT
	llm = create_llm()
	response = llm.invoke(prompt)
	print(response.content)


if __name__ == "__main__":
	main()
