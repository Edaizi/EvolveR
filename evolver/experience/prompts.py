"""
LLM Prompts for the Experience Manager
"""

# A special string to separate the natural language description from the structured part in the LLM's output.
DESCRIPTION_PART_SEPARATOR = "[DESCRIPTION]:"

STRUCTURED_PART_SEPARATOR = "[STRUCTURE]:"

SUMMARIZE_SUCCESSFUL_TRAJECTORY_PROMPT = f"""
You are an expert in analyzing interaction logs to distill generalizable wisdom.
Analyze the following successful interaction trajectory. Your goal is to extract a "Guiding Principle" from it.

A "Guiding Principle" has two parts:
1.  A concise, one-sentence natural language description. This is the core advice.
2.  A structured representation of the key steps or logic, as a list of simple (subject, predicate, object) triplets.

[Trajectory Log]:
{{trajectory_log}}

Final Outcome: SUCCESS

**Your Task:**
Based on the trajectory, generate the Guiding Principle.
First, on a new line, write `{DESCRIPTION_PART_SEPARATOR}`.
Then, write the one-sentence description of the pitfall.
Then, on a new line, write `{STRUCTURED_PART_SEPARATOR}`.
Finally, provide the structured triplets describing the failure pattern in a valid JSON list format.

[Example]:
{DESCRIPTION_PART_SEPARATOR}
When a file download fails with a 404 error, do not immediately retry the download; instead, verify the source URL's validity first.
{STRUCTURED_PART_SEPARATOR}
[
  (file download, results_in, 404 error),
  (immediate_retry, is, ineffective),
  (correct_action, is, verify URL)
]

[Output]:
"""

SUMMARIZE_FAILED_TRAJECTORY_PROMPT = f"""
You are an expert in analyzing interaction logs to find the root cause of failures.
Analyze the following failed interaction trajectory. Your goal is to extract a "Cautionary Principle" from it.

A "Cautionary Principle" has two parts:
1.  A concise, one-sentence description of the key mistake to avoid and under what circumstances.
2.  A structured representation of the failure pattern, as a list of simple (subject, predicate, object) triplets.

[Trajectory Log]:
{{trajectory_log}}

Final Outcome: FAILURE

**Your Task:**
Based on the trajectory, generate the Cautionary Principle.
First, on a new line, write `{DESCRIPTION_PART_SEPARATOR}`.
Then, write the one-sentence description of the pitfall.
Then, on a new line, write `{STRUCTURED_PART_SEPARATOR}`.
Finally, provide the structured triplets describing the failure pattern in a valid JSON list format.

[Example]:
{DESCRIPTION_PART_SEPARATOR}
When a file download fails with a 404 error, do not immediately retry the download; instead, verify the source URL's validity first.
{STRUCTURED_PART_SEPARATOR}
[
  (file download, results_in, 404 error),
  (immediate_retry, is, ineffective),
  (correct_action, is, verify URL)
]

[Output]:
"""


MATCH_PRINCIPLE_PROMPT = """
You are a semantic analysis expert. Determine if two principles describe the same core idea, even if they use different words.

Principle A: "{summary}"
Principle B: "{existing_principle_description}"

Do Principle A and Principle B describe the same essential advice or warning?
Please answer with only "Yes" or "No".
""" 

SUMMARIZE_SUCCESSFUL_TRAJECTORY_NO_STRUCTURE_PROMPT = f"""
You are an expert in analyzing interaction logs to distill generalizable wisdom.
Analyze the following successful interaction trajectory. Your goal is to extract a "Guiding Principle" from it.

A "Guiding Principle" is a concise, one-sentence natural language description of the core advice.

[Trajectory Log]:
{{trajectory_log}}

Final Outcome: SUCCESS

**Your Task:**
Based on the trajectory, generate the Guiding Principle (one-sentence description of the core advice).

[Example]:
When a file download fails with a 404 error, do not immediately retry the download; instead, verify the source URL's validity first.

[Output]:output your response here.
"""

SUMMARIZE_FAILED_TRAJECTORY_NO_STRUCTURE_PROMPT = f"""
You are an expert in analyzing interaction logs to find the root cause of failures.
Analyze the following failed interaction trajectory. Your goal is to extract a "Cautionary Principle" from it.

A "Cautionary Principle" is a concise, one-sentence description of the key mistake to avoid and under what circumstances.

[Trajectory Log]:
{{trajectory_log}}

Final Outcome: FAILURE

**Your Task:**
Based on the trajectory, generate the Cautionary Principle (one-sentence description of the pitfall).

[Example]:
When a file download fails with a 404 error, do not immediately retry the download; instead, verify the source URL's validity first.

[Output]:output your response here.
""" 