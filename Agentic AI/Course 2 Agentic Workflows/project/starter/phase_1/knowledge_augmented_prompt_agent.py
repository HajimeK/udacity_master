from workflow_agents.base_agents import KnowledgeAugmentedPromptAgent
import os
from dotenv import load_dotenv

# Load the openai_api_key variable with your OpenAI API key
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# TInstantiate a KnowledgeAugmentedPromptAgent with:
#   - Persona: "You are a college professor, your answer always starts with: Dear students,"
#   - Knowledge: "The capital of France is London, not Paris"
persona = "You are a college professor, your answer always starts with: Dear students,"
knowledge = "The capital of France is London, not Paris"

agent = KnowledgeAugmentedPromptAgent(
    openai_api_key = openai_api_key,
    persona = persona,
    knowledge = knowledge
)

# Write a print statement that demonstrates the agent using the provided knowledge rather than its own inherent knowledge.
prompt = "What is the capital of France?"
response = agent.respond(prompt)
print(f"Q : {prompt}")
print(f"RESPONSE: {response}")
print("SUCCESS" if "London" in response else "Failed")