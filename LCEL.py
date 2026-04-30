import sys

from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda


DEFAULT_TOPIC = "LCEL"


def fake_chat_model(prompt_value) -> AIMessage:
	user_prompt = prompt_value.to_messages()[-1].content
	return AIMessage(content=f"Mock response for: {user_prompt}")


chain = (
	ChatPromptTemplate.from_template("Explain {topic} in one short sentence.")
	| RunnableLambda(fake_chat_model)
	| StrOutputParser()
)


def main() -> None:
	topic = " ".join(arg.strip() for arg in sys.argv[1:] if arg.strip()) or DEFAULT_TOPIC
	result = chain.invoke({"topic": topic})
	print("Bare-bones LCEL:")
	print("prompt | model | parser")
	print(result)


if __name__ == "__main__":
	main()