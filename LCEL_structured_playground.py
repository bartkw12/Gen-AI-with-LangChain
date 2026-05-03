import sys

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from hello_world import create_llm


DEFAULT_TOPIC = "LCEL"


class TopicBreakdown(BaseModel):
	summary: str = Field(description="A short plain-English explanation of the topic.")
	key_points: list[str] = Field(description="Two or three short learning points about the topic.")
	practical_use: str = Field(description="One simple real-world use for the topic.")


def build_chain():
	prompt = ChatPromptTemplate.from_template(
		"Explain {topic} for a beginner. Return a short summary, 2-3 key points, and one practical use."
	)
	structured_llm = create_llm().with_structured_output(TopicBreakdown)
	return prompt | structured_llm


def main() -> None:
	topic = " ".join(arg.strip() for arg in sys.argv[1:] if arg.strip()) or DEFAULT_TOPIC

	try:
		chain = build_chain()
		result = chain.invoke({"topic": topic})
	except RuntimeError as exc:
		print(exc)
		print("Set Azure OpenAI credentials in environment variables. Do not hardcode keys in this file.")
		raise SystemExit(1)

	print("LCEL structured playground:")
	print("prompt | AzureChatOpenAI.with_structured_output(schema)")
	print(result.model_dump_json(indent=2))


if __name__ == "__main__":
	main()