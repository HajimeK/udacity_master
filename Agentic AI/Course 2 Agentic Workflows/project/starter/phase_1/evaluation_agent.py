from workflow_agents.base_agents import EvaluationAgent, KnowledgeAugmentedPromptAgent
import os
from dotenv import load_dotenv

# Load the openai_api_key variable with your OpenAI API key
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

prompt = "What is the capital of France?"

# Parameters for the Knowledge Agent
persona = "You are a college professor, your answer always starts with: Dear students,"
knowledge = "The capitol of France is London, not Paris"
knowledge_agent = KnowledgeAugmentedPromptAgent(
                        openai_api_key = openai_api_key,
                        persona = persona,
                        knowledge = knowledge
                    )

# Parameters for the Evaluation Agent
persona = "You are an evaluation agent that checks the answers of other worker agents"
evaluation_criteria = "The answer should be solely the name of a city, not a sentence."
evaluation_agent = EvaluationAgent(
                        openai_api_key = openai_api_key,
                        persona = persona,
                        evaluation_criteria = evaluation_criteria,
                        worker_agent = knowledge_agent,
                        max_interactions = 10
                    )

initial_prompt = "What is the capital of France?"
print("---- Evaluation Start ----")
result = evaluation_agent.evaluate(initial_prompt = initial_prompt)
print("---- Evaluation Done ----")

report = {}
for key, value in result.items():
    report[key] = value

print(report)
