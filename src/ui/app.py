import sys
from pathlib import Path
from typing import Any

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.api.v1.agents.agent import run_smart_banking_agent


st.set_page_config(
    page_title="Smart Banking Assistant",
    page_icon="🏦",
    layout="wide",
)


def display_citations(citations: list[dict[str, Any]]) -> None:
    if not citations:
        return

    with st.expander("📎 Citations"):
        for index, citation in enumerate(citations, start=1):
            st.markdown(f"**Citation {index}**")
            st.write(f"Document: {citation.get('document_name')}")
            st.write(f"Page: {citation.get('page_number')}")
            st.write(f"Chunk Type: {citation.get('chunk_type')}")
            st.write(f"Matched By: {citation.get('matched_by')}")
            st.write(f"Rerank Score: {citation.get('rerank_score')}")
            st.divider()


def display_sql_details(sql_query: str | None, sql_result: Any | None) -> None:
    if not sql_query and not sql_result:
        return

    with st.expander("🧾 SQL Details"):
        if sql_query:
            st.code(sql_query, language="sql")

        if sql_result:
            st.dataframe(sql_result, use_container_width=True)


def format_answer(response: dict[str, Any]) -> str:
    answer = response.get("answer")

    if isinstance(answer, dict):
        sql_answer = answer.get("sql_answer", "")
        document_answer = answer.get("document_answer", "")

        final_text = ""

        if sql_answer:
            final_text += f"**SQL Answer:**\n{sql_answer}\n\n"

        if document_answer:
            final_text += f"**Document Answer:**\n{document_answer}"

        return final_text.strip()

    return str(answer)


def main() -> None:
    st.title("🏦 Smart Banking Assistant")
    st.caption("Ask about banking policies, accounts, transactions, loans, FDs, and credit cards.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.sidebar:
        st.subheader("Sample Questions")

        sample_questions = [
            "hi",
            "how are you",
            "1+1",
            "What are the foreclosure charges for fixed rate home loans before 2022?",
            "Show me all active FDs for account 1345367",
            "Show international transactions on credit card CC-881001 and explain international transaction fees",
            "What is the current outstanding balance and next EMI due date for loan account L-789012?",
            "List all transactions above 50000 for account 1345367",
        ]

        selected_question = st.selectbox("Choose one:", sample_questions)

        if st.button("Ask selected question"):
            st.session_state.pending_question = selected_question

        if st.button("Clear chat"):
            st.session_state.messages = []
            st.rerun()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            response = message.get("response")
            if response:
                display_citations(response.get("citations", []))
                display_sql_details(
                    response.get("sql_query"),
                    response.get("sql_result"),
                )

    user_query = st.chat_input("Ask your banking question...")

    if "pending_question" in st.session_state:
        user_query = st.session_state.pending_question
        del st.session_state.pending_question

    if user_query:
        st.session_state.messages.append(
            {
                "role": "user",
                "content": user_query,
            }
        )

        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = run_smart_banking_agent(user_query)
                    answer_text = format_answer(response)

                    st.markdown(answer_text)
                    st.caption(f"Query Path: {response.get('query_path')}")
                    st.caption(f"Retry Count: {response.get('retry_count')}")

                    confidence = response.get("confidence_score")
                    if confidence is not None:
                        st.caption(f"Confidence Score: {round(confidence, 4)}")

                    display_citations(response.get("citations", []))
                    display_sql_details(
                        response.get("sql_query"),
                        response.get("sql_result"),
                    )

                    with st.expander("Raw JSON Response"):
                        st.json(response)

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer_text,
                            "response": response,
                        }
                    )

                except Exception as exc:
                    error_message = f"Error while processing query: {exc}"
                    st.error(error_message)

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": error_message,
                        }
                    )


if __name__ == "__main__":
    main()
