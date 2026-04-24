from __future__ import annotations

import json
import sys
from pathlib import Path

from openai import AzureOpenAI


CONFIG_PATH = Path(__file__).with_name("config_V2025_05_31.json")
DEFAULT_PROFILE = "OpenAI2"
DEFAULT_MODEL_LABEL = "GPT-5-mini"
DEFAULT_PROMPT = "Tell me a joke about trains."


def load_model_settings(profile_name: str, model_label: str) -> tuple[dict, str]:
	config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
	profile = config.get(profile_name)

	if profile is None:
		available_profiles = ", ".join(sorted(config))
		raise KeyError(f"Profile '{profile_name}' was not found. Available profiles: {available_profiles}")

	api_version = profile.get("api_version") or profile.get("version")
	deployment_name = profile.get("models", {}).get(model_label, model_label)

	missing_fields = [
		field_name
		for field_name, field_value in {
			"endpoint": profile.get("endpoint"),
			"key": profile.get("key"),
			"api_version": api_version,
		}.items()
		if not field_value
	]
	if missing_fields:
		missing = ", ".join(missing_fields)
		raise ValueError(f"Profile '{profile_name}' is missing required fields: {missing}")

	profile["api_version"] = api_version
	return profile, deployment_name


def create_client(profile_name: str, model_label: str) -> tuple[AzureOpenAI, str]:
	profile, deployment_name = load_model_settings(profile_name, model_label)
	client = AzureOpenAI(
		azure_endpoint=profile["endpoint"],
		api_key=profile["key"],
		api_version=profile["api_version"],
	)
	return client, deployment_name


def extract_message_text(content: object) -> str:
	if isinstance(content, str):
		return content.strip()

	if not isinstance(content, list):
		return ""

	text_parts: list[str] = []
	for item in content:
		text_value = getattr(item, "text", None)
		if isinstance(text_value, str) and text_value.strip():
			text_parts.append(text_value.strip())
			continue

		if isinstance(item, dict):
			raw_text = item.get("text")
			if isinstance(raw_text, str) and raw_text.strip():
				text_parts.append(raw_text.strip())

	return "\n".join(text_parts).strip()


def ask_azure_openai(client: AzureOpenAI, deployment_name: str, prompt: str) -> str:
	responses_error: Exception | None = None

	try:
		response = client.responses.create(model=deployment_name, input=prompt)
		output_text = getattr(response, "output_text", "")
		if isinstance(output_text, str) and output_text.strip():
			return output_text.strip()
	except Exception as exc:
		responses_error = exc

	try:
		response = client.chat.completions.create(
			model=deployment_name,
			messages=[{"role": "user", "content": prompt}],
		)
		message = response.choices[0].message
		reply_text = extract_message_text(message.content)
		if reply_text:
			return reply_text
	except Exception as exc:
		if responses_error is not None:
			raise RuntimeError(
				"Azure OpenAI request failed with both the Responses API and Chat Completions. "
				f"Responses error: {responses_error}. Chat error: {exc}"
			) from exc
		raise

	raise RuntimeError("Azure OpenAI returned an empty response.")


def main() -> None:
	prompt = " ".join(sys.argv[1:]).strip() or DEFAULT_PROMPT
	client, deployment_name = create_client(DEFAULT_PROFILE, DEFAULT_MODEL_LABEL)
	response_text = ask_azure_openai(client, deployment_name, prompt)
	print(response_text)


if __name__ == "__main__":
	main()
