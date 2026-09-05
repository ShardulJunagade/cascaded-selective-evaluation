system_prompt = "You are a helpful vision-language assistant."

fewshot_inst_prompt = """Given an image, an instruction, and two assistant responses to the instruction, an annotator chose which assistant's answer is more preferred. Given examples of the annotator's decision, predict the annotator's verdict on the given example. If Assistant A's response is more preferred than Assistant B's, the annotator chose "[[A]]". If Assistant B's response is more preferred than Assistant A's, the annotator chose "[[B]]"."""

fewshot_example_prompt = """
[Image]
The image for this example is shown above.

[Instruction]
{instruction}

[Assistant A's response]
{assistant_a}

[Assistant B's response]
{assistant_b}

[Preferred response]
{preferred_response}
"""

fewshot_query_prompt = """
[Image]
The image for this example is shown above.

[Instruction]
{instruction}

[Assistant A's response]
{assistant_a}

[Assistant B's response]
{assistant_b}

[Preferred response]"""
