from __future__ import annotations

import os
import sys
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import AzureChatOpenAI
from pydantic import SecretStr


DEFAULT_TOPIC = "LCEL"
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
		"Set your Azure OpenAI endpoint, API key, API version, and deployment before running the live demo."
	)


def has_azure_configuration() -> bool:
	return all(os.getenv(name, "").strip() for name in REQUIRED_ENV_VARS)


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


def fake_chat_model(prompt_value: Any) -> AIMessage:
	user_prompt = prompt_value.to_messages()[-1].content
	return AIMessage(
		content=(
			"LCEL lets you compose prompts, models, parsers, and Python functions into one runnable pipeline. "
			f"This mock model received: {user_prompt}"
		)
	)


def build_local_chain() -> RunnableLambda | Any:
	prompt = ChatPromptTemplate.from_template(
		"You are teaching a beginner. Explain what {topic} is used for in one short sentence."
	)
	return prompt | RunnableLambda(fake_chat_model) | StrOutputParser()


def build_live_chain() -> Any:
	prompt = ChatPromptTemplate.from_template(
		"You are a concise LangChain tutor. In 2 short bullet points, explain what {topic} is and one practical use for it."
	)
	return prompt | create_llm() | StrOutputParser()


def print_intro() -> None:
	print("LCEL stands for LangChain Expression Language.")
	print("It is the syntax LangChain uses to compose steps into runnable pipelines with the | operator.")
	print("Typical use: prompt -> model -> parser, or larger flows with retrieval and custom Python logic.")
	print()


def run_local_demo(topic: str) -> None:
	chain = build_local_chain()
	result = chain.invoke({"topic": topic})
	print("Local LCEL demo:")
	print(result)
	print()
	print("What this demonstrates:")
	print("- A prompt template renders the input")
	print("- A model step consumes that prompt")
	print("- An output parser normalizes the result into a plain string")


def run_live_demo(topic: str) -> None:
	if not has_azure_configuration():
		print("Live Azure demo skipped because the required environment variables are not set.")
		print("Set the Azure OpenAI variables from hello_world.py and rerun with --azure.")
		return

	chain = build_live_chain()
	result = chain.invoke({"topic": topic})
	print("Live Azure LCEL demo:")
	print(result)


def main() -> None:
	args = [arg.strip() for arg in sys.argv[1:] if arg.strip()]
	use_azure = "--azure" in args
	topic_parts = [arg for arg in args if arg != "--azure"]
	topic = " ".join(topic_parts) or DEFAULT_TOPIC

	print_intro()
	run_local_demo(topic)

	if use_azure:
		print()
		run_live_demo(topic)
	else:
		print()
		print("Tip: run `python LCEL.py --azure` to swap the mock model for AzureChatOpenAI.")


if __name__ == "__main__":
	main()