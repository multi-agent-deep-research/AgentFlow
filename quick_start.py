# Import the solver
from agentflow.agentflow.solver import construct_solver

# Set the LLM engine name
llm_engine_name = "dashscope" # you can use "gpt-4o" as well
# llm_engine_name = "gpt-4o"

# Construct the solver
solver = construct_solver(llm_engine_name=llm_engine_name)

# Solve the user query
output = solver.solve("What is the capital of France?")
print(output["direct_output"])