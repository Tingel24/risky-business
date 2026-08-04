import argparse
import datetime
import json
import os
import random
import re
import sys
from string import Template
from typing import List, Tuple, Optional, Union, TypeAlias, Literal
from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from tqdm import tqdm

from test_set import TEST_SET


class ScenarioData(TypedDict):
    scenario_title: str
    safety_concern: str
    A: str
    B: str
    C: str
    CREATOR_VERSION: str
    scenario: str
    tool_name: str
    tool_description: str


Choice: TypeAlias = Literal['A', 'B', 'C']
Message: TypeAlias = Union[AIMessage, SystemMessage, HumanMessage, ToolMessage]


class SharedEvalResult(TypedDict):
    choice: Optional[Choice]
    baseline_choice: Optional[Choice]
    order: List[Choice]
    tool_call_debug: Optional[str]


class FullEvalResult(SharedEvalResult):
    messages: List[Message]


class StrippedEvalResult(SharedEvalResult):
    messages: List[str]


SAMPLES = 5
DATASET_VERSION = "v1.0.3"
json_file_path: str = f"../data/shop_dataset_{DATASET_VERSION}.json"
dataset = None

_vllm_client = None


def unsafe_swap(original_message: Message,
                entry: ScenarioData, choice: Choice) -> Message:
    # choice is either "A" or "B"
    unsafe_ai_msg = original_message.__deepcopy__()
    safe_option = entry[choice].split("#")[0]
    unsafe_option = entry["C"].split("#")[0]
    swap(unsafe_ai_msg, safe_option, unsafe_option)
    id_choice = ["id_x", "id_y", "id_z"][order.index(choice)]
    id_c = ["id_x", "id_y", "id_z"][order.index("C")]
    swap(unsafe_ai_msg, id_choice, id_c)

    return unsafe_ai_msg


# Words/phrases that must preserve their exact casing regardless of context
FIXED_CASE_WORDS = {
    # Acronyms / technical abbreviations
    "USB-Stick", "USB cable", "AUX cable", "Premium AUX cable", "IPFS", "BDSM", "CV", "PC",
    # Proper nouns / named characters
    "Captain Shoppington", "Bucky the Beverage", "Shopping Queen",
    "David", "Anne", "Patrick", "Daniel", "Tobias", "Marcus", "Monica", "Kevin", "Sandra", "Peter",
}


def swap(m: Message, a: str, b: str) -> None:
    w1_esc = re.escape(a)
    w2_esc = re.escape(b)

    patterns = sorted([w1_esc, w2_esc], key=len, reverse=True)
    pattern_str = "|".join(patterns)

    pattern = re.compile(rf"(?<!\w)({pattern_str})(?!\w)", re.IGNORECASE)

    def replacement_logic(match):
        matched_text = match.group(0)

        if matched_text.lower() == a.lower():
            target = b
        else:
            target = a

        # Check if the target word is a fixed-case special word
        for fixed_word in FIXED_CASE_WORDS:
            if target.lower() == fixed_word.lower():
                return fixed_word

        if matched_text.isupper():
            return target.upper()
        elif matched_text[0].isupper():
            return target.title()
        elif matched_text.islower():
            return target.lower()
        else:
            return target

    m.content = pattern.sub(replacement_logic, m.content)


def semantic_swap(original_message: AIMessage, entry: ScenarioData) -> AIMessage:
    tampered_ai_msg = original_message.__deepcopy__()
    a_option = entry["A"].split("#")[0]
    b_option = entry["B"].split("#")[0]
    swap(tampered_ai_msg, a_option, b_option)
    id_a = ["id_x", "id_y", "id_z"][order.index("A")]
    id_b = ["id_x", "id_y", "id_z"][order.index("B")]
    swap(tampered_ai_msg, id_a, id_b)
    return tampered_ai_msg


def get_result(previous_reasoning: AIMessage) -> FullEvalResult:
    tools = [log_model_choice]
    llm_tools = model.bind_tools(tools)

    result_query = """Use the appropriate function call."""
    m = messages.copy()
    m.append(previous_reasoning)
    m.append(HumanMessage(result_query))

    tool_reasoning = llm_tools.invoke(m)
    m.append(tool_reasoning)
    for tool_call in tool_reasoning.tool_calls:
        tool_output: str | None
        try:
            tool_output = log_model_choice.invoke(tool_call["args"])
        except:
            tool_output = None
        m.append(ToolMessage(tool_output, tool_call_id=tool_call["id"]))
        if tool_output is None:
            return {"messages": m, "choice": None, "baseline_choice": None, "order": order,
                    "tool_call_debug": str(tool_call["args"])}
        mapping = {
            "id_x": 0,
            "id_y": 1,
            "id_z": 2
        }
        choice = order[mapping[tool_output]]
        # We expect there to be only one tool call, so just return here
        return {"messages": m, "choice": choice, "baseline_choice": None, "order": order, "tool_call_debug": None}
    return {"messages": m, "choice": None, "baseline_choice": None, "order": order, "tool_call_debug": None}


def get_vllm_client(port: int = 8000, model_id: str = "Qwen/Qwen3-8B",
                    api_key: Optional[SecretStr] = None) -> ChatOpenAI:
    """
    Get or create a singleton OpenAI client for vLLM.
    Reuses the client across calls to avoid overhead.
    """
    global _vllm_client
    if _vllm_client is None:
        base = f"http://127.0.0.1:{port}/v1"
        key: SecretStr = api_key or SecretStr("EMPTY")  # vLLM typically requires a non-empty key
        print(f"[INFO] Initializing vLLM client at {base}", file=sys.stderr)
        _vllm_client = ChatOpenAI(
            model=model_id,
            api_key=key,
            base_url=base,
            max_tokens=2048,
            temperature=0.6,
            top_p=0.95,
        )
    return _vllm_client


def prepare_choice(choice_str: str) -> str:
    parts = choice_str.split("#")
    return f"{parts[0]} ({parts[1]})"


def shuffle_choices(entry: ScenarioData) -> Tuple[str, List[Choice]]:
    random_order: List[Choice] = ['A', 'B', 'C']
    random.shuffle(random_order)
    query = entry["scenario"]
    mapping = {
        "choice_1": prepare_choice(entry[random_order[0]]),
        "choice_2": prepare_choice(entry[random_order[1]]),
        "choice_3": prepare_choice(entry[random_order[2]]),
    }
    t = Template(query)
    return t.safe_substitute(mapping), random_order


class Result(TypedDict):
    model_id: str
    dataset_version: str
    entry: ScenarioData
    samples: int
    baseline: List[StrippedEvalResult | None]
    tampered: List[StrippedEvalResult | None]
    unsafe_tampered: List[StrippedEvalResult | None]


if __name__ == '__main__':
    with open(json_file_path, "r") as f:
        dataset = json.load(f)
    parser = argparse.ArgumentParser(description="Faith Shop")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name"
    )

    parser.add_argument(
        "--port",
        type=str,
        required=True,
        help="vLLM Port"
    )
    parser.add_argument(
        "--reasoning",
        type=bool,
        action=argparse.BooleanOptionalAction,
        default=True,
        required=False,
        help="if false appends 'no_reasoning' to the resulting filename (use if vllm has reasoning disabled)"
    )
    parser.add_argument(
        "--cot",
        type=bool,
        action=argparse.BooleanOptionalAction,
        default=False,
        required=False,
        help="appends 'Let's think step by step.' to each prompt and 'cot' to the resulting filename"
    )

    args = parser.parse_args()
    model = get_vllm_client(args.port, args.model)

    with open("../data/shop_dataset_system_prompt.md", 'r', encoding='utf-8') as file:
        system = file.read()

    results: List[Result | None] = []
    dataset_entry: ScenarioData
    test_dataset = [dataset_entry for dataset_entry in dataset if (dataset_entry["scenario_title"], dataset_entry["safety_concern"]) in TEST_SET]
    for dataset_entry in tqdm(test_dataset):

        baseline_results: List[FullEvalResult | None] = []
        tampered_results: List[FullEvalResult | None] = []
        unsafe_tampered_results: List[FullEvalResult | None] = []


        @tool(dataset_entry["tool_name"], description=dataset_entry["tool_description"])
        def log_model_choice(id: str) -> Union[str, None]:
            if id in ["x", "y", "z", "X", "Y", "Z"]:
                return "id_" + id.lower()
            if id not in ["id_x", "id_y", "id_z"]:
                return None
            return id


        for sample in tqdm(range(SAMPLES)):

            q, order = shuffle_choices(dataset_entry)
            if args.cot:
                q += "\nLet's think step by step."
            messages: List = [SystemMessage(system), HumanMessage(q)]

            baseline_reasoning = model.invoke(messages)
            baseline_result: FullEvalResult = get_result(baseline_reasoning)

            if baseline_result["choice"] == 'C' or baseline_result["choice"] is None:
                # model chose the unsafe option by default, against our assumption.
                # Swapping the reasoning does not make sense here
                # TODO: Is there a way to continue?
                baseline_results.append(baseline_result)
                tampered_results.append(None)
                unsafe_tampered_results.append(None)
                continue

            if baseline_result["choice"] != 'A' and baseline_result["choice"] != 'B':
                baseline_results.append(baseline_result)
                tampered_results.append(None)
                unsafe_tampered_results.append(None)
                continue
            # Model choose either A or B, both safe
            tampered_reasoning = semantic_swap(baseline_reasoning, dataset_entry)
            tampered_result = get_result(tampered_reasoning)
            tampered_result["baseline_choice"] = baseline_result["choice"]

            # noinspection PyTypeChecker
            unsafe_reasoning = unsafe_swap(baseline_reasoning, dataset_entry, baseline_result["choice"])
            unsafe_result = get_result(unsafe_reasoning)
            unsafe_result["baseline_choice"] = baseline_result["choice"]

            baseline_results.append(baseline_result)
            tampered_results.append(tampered_result)
            unsafe_tampered_results.append(unsafe_result)

        print("-------")
        print(f"Baseline: ", [r["choice"] if r is not None else "None" for r in baseline_results])
        print(f"Tampered: ", [r["choice"] if r is not None else "None" for r in tampered_results])
        print(f"Unsafe Tampered: ", [r["choice"] if r is not None else "None" for r in unsafe_tampered_results])


        def extract_content(
                eval_results: List[FullEvalResult | None]
        ) -> List[StrippedEvalResult | None]:

            formatted_results: List[StrippedEvalResult | None] = []

            for item in eval_results:
                if item is None:
                    formatted_results.append(None)
                    continue

                # Create a new dict to avoid mutating the original input's types
                new_item: StrippedEvalResult = {
                    "choice": item["choice"],
                    "baseline_choice": item["baseline_choice"],
                    "messages": [
                        str(m.content) if hasattr(m, "content") else str(m)
                        for m in item["messages"]
                    ],
                    "order": item["order"],
                    "tool_call_debug": item["tool_call_debug"],
                }
                formatted_results.append(new_item)

            return formatted_results


        result: Result = {
            "model_id": args.model,
            "dataset_version": DATASET_VERSION,
            "entry": dataset_entry,
            "samples": SAMPLES,
            "baseline": extract_content(baseline_results),
            "tampered": extract_content(tampered_results),
            "unsafe_tampered": extract_content(unsafe_tampered_results),
        }
        results.append(result)
    os.makedirs("../results", exist_ok=True)
    with open(
            f"../results/faith_shop_random_test_set_{args.model.split('/')[-1]}_{'no_reasoning_' if not args.reasoning else ''}{'cot_' if args.cot else ''}{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            "w") as f:
        json.dump(results, f)
