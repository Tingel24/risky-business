import difflib
import json
from typing import List

import streamlit as st

import faith_shop


@st.cache_data
def load_data() -> List[dict]:
    with open('../results/faith_shop_QwQ-32B_20260504_221900.json', 'r') as f:
        return json.load(f)


def highlight_diff(base_text, current_text):
    """Highlights differences between base_text and current_text."""
    if base_text == current_text:
        return current_text

    base_words = base_text.split()
    curr_words = current_text.split()

    matcher = difflib.SequenceMatcher(None, base_words, curr_words)
    output = []

    for opcode, a0, a1, b0, b1 in matcher.get_opcodes():
        if opcode == 'equal':
            output.append(" ".join(curr_words[b0:b1]))
        elif opcode in ['replace', 'insert']:
            seq = " ".join(curr_words[b0:b1])
            output.append(
                f'<mark style="background-color: #ffff00; color: black; padding: 0 2px; border-radius: 2px;">{seq}</mark>'
            )

    return " ".join(output)


def filter_results(x) -> bool:
    if x is None:
        return False
    if not x.get("baseline"):
        return False
    base = x["baseline"][0]
    if base is None:
        return False
    return base.get("choice") == "C"


# --- DATA LOADING AND FILTERING ---
raw_data = load_data()
# data = list(filter(filter_results, raw_data))
data = raw_data
if not data:
    st.error("No data matches the filter criteria.")
    st.stop()

# --- SESSION STATE INITIALIZATION ---
if 'index' not in st.session_state:
    st.session_state.index = 0

if 'eval_mode' not in st.session_state:
    st.session_state.eval_mode = "baseline"

# Ensure index remains valid after filtering
if st.session_state.index >= len(data):
    st.session_state.index = 0

# --- CSS ---
st.markdown("""
    <style>
    .small-font { font-size: 13px !important; color: #555555; line-height: 1.4; }
    .reasoning-text { font-size: 13px !important; font-style: italic; }
    mark { border-bottom: 2px solid orange; }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR ---
st.sidebar.title("Navigation")
col1, col2 = st.sidebar.columns(2)

if col1.button("Previous") and st.session_state.index > 0:
    st.session_state.index -= 1
    st.rerun()

if col2.button("Next") and st.session_state.index < len(data) - 1:
    st.session_state.index += 1
    st.rerun()

st.sidebar.markdown(f"**Entry:** {st.session_state.index + 1} / {len(data)}")
st.sidebar.divider()

modes = ["baseline", "tampered", "unsafe_tampered"]
st.sidebar.radio(
    "Select Evaluation Mode:",
    modes,
    key="eval_mode"
)

# --- MAIN UI ---
current_data = data[st.session_state.index]
entry_meta = current_data.get('entry', {})

st.title(entry_meta.get('scenario_title', 'Unknown Title'))
st.subheader(f"Concern: {entry_meta.get('safety_concern', 'None')}")

with st.expander("View Scenario Context", expanded=True):
    choices = faith_shop.shuffle_choices(entry_meta)
    if choices:
        st.text(choices[0])

st.divider()

# MESSAGE PROCESSING
mode_title = st.session_state.eval_mode.replace('_', ' ').title()
st.subheader(f"Chat History: {mode_title}")

current_mode_list = current_data.get(st.session_state.eval_mode, [])
baseline_list = current_data.get("baseline", [])

if not current_mode_list or current_mode_list[0] is None:
    st.warning("Data is missing or null for the selected evaluation mode.")
elif not baseline_list or baseline_list[0] is None:
    st.warning("Baseline data is missing or null.")
else:
    curr_msgs = current_mode_list[0].get('messages', [])
    base_msgs = baseline_list[0].get('messages', [])

    if not curr_msgs:
        st.warning("No messages found in the current mode.")
    else:
        for i, msg in enumerate(curr_msgs):
            # Determine Role
            if i == 0:
                role = "system"
            elif i % 2 != 0:
                role = "user"
            else:
                role = "assistant"

            should_diff = (st.session_state.eval_mode != "baseline" and i == 2)

            if should_diff and i < len(base_msgs):
                display_text = highlight_diff(base_msgs[i], msg)
            else:
                display_text = msg

            with st.chat_message(role):
                if role == "system":
                    with st.expander("Show System Prompt", expanded=False):
                        st.markdown(f'<div class="small-font">{display_text}</div>', unsafe_allow_html=True)

                elif role == "assistant":
                    if "<think>" in display_text:
                        parts = display_text.split("</think>")
                        think_content = parts[0].replace("<think>", "").strip()
                        with st.status("Reasoning Process...", expanded=False):
                            st.markdown(f'<div class="reasoning-text">{think_content}</div>', unsafe_allow_html=True)
                        if len(parts) > 1:
                            st.markdown(parts[1].strip(), unsafe_allow_html=True)
                    else:
                        st.markdown(display_text, unsafe_allow_html=True)
                else:
                    st.markdown(display_text, unsafe_allow_html=True)
