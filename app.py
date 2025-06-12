import os
import re
import json
import random
import pandas as pd
import streamlit as st
import openai

# ───────────────────────────────────────────────────────────────────────────────
# 1) Load your OpenAI API key from the environment
# ───────────────────────────────────────────────────────────────────────────────
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    st.error("Please set the OPENAI_API_KEY environment variable and restart.")
    st.stop()

# ───────────────────────────────────────────────────────────────────────────────
# 2) Load catalog.json and Labelled Expressions.xlsx
# ───────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_catalog(path="catalog.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_data
def load_labelled(path="Labelled Expressions.xlsx"):
    df = pd.read_excel(path, engine="openpyxl")
    assert "NLP Description" in df.columns
    assert "Expression" in df.columns
    return df

catalog       = load_catalog()
labelled_expr = load_labelled()

# ───────────────────────────────────────────────────────────────────────────────
# 3) Few-shot helpers (keyword‐overlap selection)
# ───────────────────────────────────────────────────────────────────────────────
def pick_few_shot_examples(query: str, df: pd.DataFrame, k: int = 6) -> pd.DataFrame:
    q_words = set(re.findall(r"\w+", query.lower()))
    scores  = []
    for idx, row in df.iterrows():
        desc       = row["NLP Description"]
        desc_words = set(re.findall(r"\w+", desc.lower()))
        overlap    = len(q_words & desc_words)
        if overlap:
            scores.append((overlap, idx))
    scores.sort(key=lambda x: x[0], reverse=True)
    top_idxs = [idx for _, idx in scores[:k]]
    if len(top_idxs) < k:
        remaining = set(df.index) - set(top_idxs)
        extra     = random.sample(list(remaining), k - len(top_idxs))
        top_idxs += extra
    return df.loc[top_idxs]

def build_few_shot_prompt(query: str, df_examples: pd.DataFrame, k: int = 6) -> str:
    few_shot = pick_few_shot_examples(query, df_examples, k=k)
    lines = []
    for _, row in few_shot.iterrows():
        nl_desc = row["NLP Description"].strip()
        expr    = row["Expression"].strip()
        lines.append(f"NLP: {nl_desc}\nCentricExpr: {expr}\n")
    lines.append(f"NLP: {query}\nCentricExpr:")
    return "\n".join(lines)

# ───────────────────────────────────────────────────────────────────────────────
# 4) The core function: NL → CentricExpr via few-shot + new API syntax
# ───────────────────────────────────────────────────────────────────────────────
def nl2centricexpr_plain_few_shot(
    query: str,
    model: str = "gpt-3.5-turbo",
    few_shot_k: int = 6
) -> str:
    prompt_text = build_few_shot_prompt(query, labelled_expr, k=few_shot_k)

    system_message = (
                "You are CentricExpr-GPT, your job is to translate natural language into centric expression. When given a line beginning with 'NLP:', output exactly one "
                "CentricExpr function call (no backticks, no extra text)"
    )

    # *** Updated call for openai>=1.0.0 ***
    resp = openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user",   "content": prompt_text}
        ],
        temperature=0.0,
        max_tokens=128,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )

    # Extract the assistant’s content
    content = resp.choices[0].message.content.strip()
    # Strip triple backticks if present
    if content.startswith("```") and content.endswith("```"):
        content = content.strip("`\n ")
    return content

# ───────────────────────────────────────────────────────────────────────────────
# 5) Streamlit UI
# ───────────────────────────────────────────────────────────────────────────────
st.title("CentricExpr Generator")
st.write("Type a natural-language query below, then click the button to get a CentricExpr expression.")

query_input = st.text_area("Natural Language Query:", height=120)
submit      = st.button("Generate CentricExpr")

# Sidebar controls
st.sidebar.header("Settings")
model_choice = st.sidebar.selectbox("Model", ["gpt-3.5-turbo", "gpt-4o-mini"], index=0)
few_k        = st.sidebar.slider("Few-Shot Examples (K)", min_value=1, max_value=10, value=6)

if submit:
    user_q = query_input.strip()
    if not user_q:
        st.warning("Please enter a query.")
    else:
        with st.spinner("Generating CentricExpr…"):
            try:
                result = nl2centricexpr_plain_few_shot(
                    user_q,
                    model=model_choice,
                    few_shot_k=few_k
                )
                st.success("👇 Generated CentricExpr:")
                st.code(result, language="plaintext")
            except Exception as e:
                st.error(f"Error generating CentricExpr:\n{e}")
