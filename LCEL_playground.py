import sys

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from hello_world import create_llm


DEFAULT_TOPIC = "LCEL"


def build_chain():
	prompt = ChatPromptTemplate.from_template("Explain {topic} in one short paragraph.")
	return prompt | create_llm() | StrOutputParser()


def main() -> None:
	topic = " ".join(arg.strip() for arg in sys.argv[1:] if arg.strip()) or DEFAULT_TOPIC

	try:
		chain = build_chain()
		result = chain.invoke({"topic": topic})
	except RuntimeError as exc:
		print(exc)
		print("Set Azure OpenAI credentials in environment variables. Do not hardcode keys in this file.")
		raise SystemExit(1)

	print("LCEL playground:")
	print("prompt | AzureChatOpenAI | parser")
	print(result)


if __name__ == "__main__":
	main()