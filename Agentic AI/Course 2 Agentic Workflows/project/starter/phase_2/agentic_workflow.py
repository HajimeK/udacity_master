# agentic_workflow.py
# Import the following agents: ActionPlanningAgent, KnowledgeAugmentedPromptAgent, EvaluationAgent, RoutingAgent from the workflow_agents.base_agents module
from workflow_agents.base_agents import ActionPlanningAgent, KnowledgeAugmentedPromptAgent, EvaluationAgent, RoutingAgent
import os
from dotenv import load_dotenv

# Load the openai_api_key variable with your OpenAI API key
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# load the product spec
# Load the product spec document Product-Spec-Email-Router.txt
# into a variable called product_spec
path_to_document_file = os.path.join(os.path.abspath("."), "Product-Spec-Emai-Router.txt")
product_spec = open(path_to_key_file, "rt").read()

# Instantiate all the agents

# Action Planning Agent
knowledge_action_planning = (
    "Stories are defined from a product spec by identifying a "
    "persona, an action, and a desired outcome for each story. "
    "Each story represents a specific functionality of the product "
    "described in the specification."
    "Features are defined by grouping related user stories."
    "Tasks are defined for each story and represent the engineering "
    "work required to develop the product."
    "A development Plan for a product contains all these components"
)
# Instantiate an action_planning_agent using the 'knowledge_action_planning'
action_planning_agent = ActionPlanningAgent(
    openai_api_key = openai_api_key,
    knowledge = knowledge_action_planning)

# Product Manager - Knowledge Augmented Prompt Agent
persona_product_manager = "You are a Product Manager, you are responsible for defining the user stories for a product."
knowledge_product_manager = (
    "Stories are defined by writing sentences with a persona, an action, and a desired outcome. "
    "The sentences always start with: As a "
    "Write several stories for the product spec below, where the personas are the different users of the product. "
    f"{product_spec}"
)

# Instantiate a product_manager_knowledge_agent using 'persona_product_manager'
# and the completed 'knowledge_product_manager'
product_manager_knowledge_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key = openai_api_key,
    persona = persona_product_manager,
    knowledge = knowledge_product_manager
)

# Product Manager - Evaluation Agent
# Define the persona and evaluation criteria for a Product Manager evaluation agent
# and instantiate it as product_manager_evaluation_agent.
# This agent will evaluate the product_manager_knowledge_agent.
# The evaluation_criteria should specify the expected structure for user stories
# (e.g., "As a [type of user], I want [an action or feature] so that [benefit/value].").

persona_product_manager_evaluation_agent = "You are an evaluation agent that checks the answers of other worker agents"
evaluation_criteria_product_manager_evaluatoin_agent = "The answer should be stories that follow the following structure: As a [type of user], I want [an action or feature] so that [benefit/value]."
product_manager_evaluation_agent = EvaluationAgent(
    openai_api_key = openai_api_key,
    persona = persona_product_manager_evaluation_agent,
    evaluation_criteria = evaluation_criteria_product_manager_evaluatoin_agent,
    worker_agent = product_manager_knowledge_agent,
    max_interactions=3
)

# Program Manager - Knowledge Augmented Prompt Agent
persona_program_manager = "You are a Program Manager, you are responsible for defining the features for a product."
knowledge_program_manager = "Features of a product are defined by organizing similar user stories into cohesive groups."
# Instantiate a program_manager_knowledge_agent using 'persona_program_manager' and 'knowledge_program_manager'
program_manager_knowledge_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key= openai_api_key,
    persona = persona_program_manager,
    knowledge = knowledge_program_manager
)
# Program Manager - Evaluation Agent

# Instantiate a program_manager_evaluation_agent using 'persona_program_manager_eval' and the evaluation criteria below.
#                      "The answer should be product features that follow the following structure: " \
#                      "Feature Name: A clear, concise title that identifies the capability\n" \
#                      "Description: A brief explanation of what the feature does and its purpose\n" \
#                      "Key Functionality: The specific capabilities or actions the feature provides\n" \
#                      "User Benefit: How this feature creates value for the user"
# For the 'agent_to_evaluate' parameter, refer to the provided solution code's pattern.
persona_program_manager_eval = "You are an evaluation agent that checks the answers of other worker agents."
evaluation_criteria_program_manager_evaluation_agent = """
The answer should be product features that follow the following structure:

Feature Name: A clear, concise title that identifies the capability
Description: A brief explanation of what the feature does and its purpose
Key Functionality: The specific capabilities or actions the feature provides
User Benefit: How this feature creates value for the user
"""
program_manager_evaluation_agent = EvaluationAgent(
    openai_api_key = openai_api_key,
    persona = persona_program_manager_eval,
    evaluation_criteria = evaluation_criteria_program_manager_evaluation_agent,
    worker_agent = program_manager_knowledge_agent,
    max_interactions = 3
)

# Development Engineer - Knowledge Augmented Prompt Agent
persona_dev_engineer = "You are a Development Engineer, you are responsible for defining the development tasks for a product."
knowledge_dev_engineer = "Development tasks are defined by identifying what needs to be built to implement each user story."
# Instantiate a development_engineer_knowledge_agent using 'persona_dev_engineer' and 'knowledge_dev_engineer'
development_engineer_knowledge_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key = openai_api_key,
    persona = persona_dev_engineer,
    knowledge = knowledge_dev_engineer,
)

# Development Engineer - Evaluation Agent
persona_dev_engineer_eval = "You are an evaluation agent that checks the answers of other worker agents."
evaluation_criteria_development_engineer_evaluation_agent = """
The answer should be tasks following this exact structure:

Task ID: A unique identifier for tracking purposes
Task Title: Brief description of the specific development work
Related User Story: Reference to the parent user story
Description: Detailed explanation of the technical work required
Acceptance Criteria: Specific requirements that must be met for completion
Estimated Effort: Time or complexity estimation
Dependencies: Any tasks that must be completed first
"""
print(evaluation_criteria_development_engineer_evaluation_agent)
# Instantiate a development_engineer_evaluation_agent using 'persona_dev_engineer_eval' and the evaluation criteria below.
#                      "The answer should be tasks following this exact structure: " \
#                      "Task ID: A unique identifier for tracking purposes\n" \
#                      "Task Title: Brief description of the specific development work\n" \
#                      "Related User Story: Reference to the parent user story\n" \
#                      "Description: Detailed explanation of the technical work required\n" \
#                      "Acceptance Criteria: Specific requirements that must be met for completion\n" \
#                      "Estimated Effort: Time or complexity estimation\n" \
#                      "Dependencies: Any tasks that must be completed first"
# For the 'agent_to_evaluate' parameter, refer to the provided solution code's pattern.
development_engineer_evaluation_agent = EvaluationAgent(
    openai_api_key = openai_api_key,
    persona = persona_dev_engineer_eval,
    evaluation_criteria = evaluation_criteria_development_engineer_evaluation_agent,
    worker_agent = development_engineer_knowledge_agent,
    max_interactions = 3
)

# Routing Agent
# Instantiate a routing_agent. You will need to define a list of agent dictionaries (routes) for Product Manager, Program Manager, and Development Engineer. Each dictionary should contain 'name', 'description', and 'func' (linking to a support function). Assign this list to the routing_agent's 'agents' attribute.
routing_agent = RoutingAgent(
    openai_api_key=openai_api_key,
)

# Job function persona support functions
# Define the support functions for the routes of the routing agent (e.g., product_manager_support_function, program_manager_support_function, development_engineer_support_function).
# Each support function should:
#   1. Take the input query (e.g., a step from the action plan).
#   2. Get a response from the respective Knowledge Augmented Prompt Agent.
#   3. Have the response evaluated by the corresponding Evaluation Agent.
#   4. Return the final validated response.
# Updated for reusability of the code
def support_function(query: str,
                    knowledge_agent: KnowledgeAugmentedPromptAgent,
                    eval_agent: EvaluationAgent):
    initial_response = knowledge_agent.respond(query)
    evaluation_response = eval_agent.evaluate(initial_response)
    return evaluation_response["final_response"]

routing_agent.agents = [
    {
        "name": "Product Manager",
        "description": "Responsible for defining product personas and user stories only. Does not define features or tasks. Does not group stories",
        "func": lambda query: support_function(query, product_manager_knowledge_agent, product_manager_evaluation_agent)
    },
    {
        "name": "Program Manager",
        "description": "Responsible for defining producct features and user benefits. Does not define tasks for imiplementation",
        "func": lambda query: support_function(query, program_manager_knowledge_agent, program_manager_evaluation_agent)
    },
    {
        "name": "Development Engineer",
        "description": "responsible for defining the development tasks for a product.",
        "func": lambda query: support_function(query, development_engineer_knowledge_agent, development_engineer_evaluation_agent)
    }
]

# Run the workflow
print("\n*** Workflow execution started ***\n")
# Workflow Prompt
# ****
workflow_prompt = "What would the development tasks for this product be?"
# ****
print(f"Task to complete in this workflow, workflow prompt = {workflow_prompt}")

print("\nDefining workflow steps from the workflow prompt")
# Implement the workflow.
#   1. Use the 'action_planning_agent' to extract steps from the 'workflow_prompt'.
steps = action_planning_agent.extract_steps_from_prompt(workflow_prompt)

#   2. Initialize an empty list to store 'completed_steps'.
#   3. Loop through the extracted workflow steps:
#      a. For each step, use the 'routing_agent' to route the step to the appropriate support function.
#      b. Append the result to 'completed_steps'.
#      c. Print information about the step being executed and its result.
def exec_step(step: str):
    print(f"step: {step}")
    result = routing_agent.route(step)
    print(f"result: {result}")
    return result

completed_steps = [exec_step(step) for step in steps]

#   4. After the loop, print the final output of the workflow (the last completed step).
print(f"~~~~~~~~~~~~~~~~~~~\nfinal output\n~~~~~~~~~~~~~~~~~~~\n ")
for step in completed_steps:
    print(step + "\n\n")