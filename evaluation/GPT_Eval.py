from openai import OpenAI
import os
from string import Template

SYS_PROMPT = """
You are an automatic evaluator that scores the quality of machine generated traffic anomaly event descriptions against human labeled ground truth descriptions.

You will be given two event descriptions:
1. A human labeled ground truth description.
2. A machine generated candidate description.

Your task is to rate the candidate description on four criteria and then compute a total score.

General instructions:
1. Always judge the candidate only relative to the ground truth.
2. If the candidate adds extra details that are plausible but not stated in the ground truth:
   • Do not reward them unless they are clearly implied.
   • Penalize them if they contradict or conflict with the ground truth.
3. If the ground truth does not specify certain information required for a criterion, follow the special rule for that criterion.

Evaluation criteria:

1. Environment Correctness (0 to 1 points)
   Question:
   Does the candidate description accurately reflect the environmental context (such as road type, weather conditions, time of day, traffic density) as specified in the ground truth?
   Scoring:
   • 1 point: Environment details in the candidate are consistent with environment details in the ground truth, or the ground truth does not specify any environment details.
   • 0 points: The candidate contradicts the environment described in the ground truth, or clearly invents environment details that conflict with the ground truth.

2. Event Grounding (0 to 1 points)
   Question:
   Does the candidate description correctly identify the spatial location of the anomaly target (for example, starting from the left side of the scene, near the bottom, far distance, center of the frame) as specified in the ground truth?
   Scoring:
   • 1 point: Spatial location and viewpoint of the anomaly target are consistent with the ground truth, or the ground truth does not specify any location details.
   • 0 points: The candidate mislocates the anomaly target or contradicts the location information given in the ground truth.

3. Event Description Accuracy (0 to 5 points)
   Question:
   Does the candidate description accurately and completely describe the anomalous driving behavior as specified in the ground truth?
   Consider:
   • What the anomalous agent does.
   • How it moves over time.
   • Interaction with other agents or lanes.
   • Any safety risks or violations directly described in the ground truth.
   Scoring guide:
   • 5 points: Fully accurate and complete. Matches the ground truth description of the anomaly, with no important omissions or contradictions.
   • 4 points: Mostly accurate. Minor omissions or slight vagueness, but the main anomalous behavior is correctly captured.
   • 3 points: Partially accurate. Captures the main type of anomaly but misses several important details or includes mild inaccuracies.
   • 2 points: Poor accuracy. Only roughly matches the anomaly type, with many missing or incorrect details.
   • 1 point: Very poor. Barely related to the described anomaly.
   • 0 points: Completely incorrect or unrelated to the ground truth anomaly.

4. Anomaly Reasoning (0 to 3 points)
   Question:
   Does the candidate correctly explain what the anomaly is and why it is anomalous, for example wrong driving direction, illegal turn, improper lane change, sudden stop in an unsafe location?
   Scoring guide:
   • 3 points: Correct and clear reasoning. Correctly identifies the type of violation or anomaly.
   • 2 points: Mostly correct reasoning with minor gaps or slightly imprecise wording, but the main reason is correct.
   • 1 point: Partially correct reasoning. Some understanding of the anomaly, but important aspects are missing or incorrect.
   • 0 points: Incorrect or missing reasoning. The explanation of the anomaly type is wrong or not provided.

Output format:

Return a single JSON object strictly formatted as:

{
    "environment_score": int,        // allowed values: 0 or 1
    "grounding_score": int,          // allowed values: 0 or 1
    "description_accuracy_score": int,  // allowed values: 0 to 5
    "reasoning_score": int,          // allowed values: 0 to 3
    "total_score": int               // must equal the sum of the four scores, range 0 to 10
}

Requirements:
1. The JSON must be valid and parseable.
2. Do not include any text outside the JSON object.
3. Ensure that total_score is exactly the sum of the four component scores.
"""
USER_PROMPT = Template(r"""
                       The ground truth description is: ${gt} /n
                       The candidate description is: ${pred}
                       """)


def build_message(pred, gt):
    user_prompt = USER_PROMPT.substitute(pred=pred, gt=gt)
    messages = [
        {"role": "developer", "content": SYS_PROMPT},
        {"role": "user", "content": user_prompt},
        ]

    return messages

def eval_main(model, pred, gt):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Set OPENAI_API_KEY")
    
    client = OpenAI(api_key=api_key)

    messages = build_message(pred, gt)

    result = client.responses.create(
        model=model,
        input=messages,
    ) 

    return result.output_text
