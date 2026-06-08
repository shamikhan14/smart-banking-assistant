import os
import re
from decimal import Decimal
from datetime import date, datetime
from typing import Any
from uuid import UUID
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from src.core.db import get_db_conn

load_dotenv()

OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")


# ---------------------------------------------------------------------------
# Banking schema context
# ---------------------------------------------------------------------------

BANKING_SCHEMA_CONTEXT = """
You are working with a PostgreSQL banking database.

Allowed tables and columns:

1. accounts
- account_id VARCHAR
- customer_name
- account_type
- branch_code
- ifsc_code
- mobile
- email
- kyc_status
- created_at

2. transactions
- txn_id
- account_id VARCHAR
- txn_date
- txn_type: only 'debit' or 'credit'
- amount
- balance_after
- description
- channel
- merchant_name
- category
- created_at

3. loan_accounts
- loan_id VARCHAR
- account_id VARCHAR
- loan_type
- principal
- outstanding
- disbursed_date
- emi_amount
- next_emi_date
- interest_rate
- tenure_months
- emi_paid
- status
- created_at

4. fixed_deposits
- fd_id VARCHAR
- account_id VARCHAR
- principal
- interest_rate
- tenure_days
- start_date
- maturity_date
- maturity_amount
- interest_payout
- status
- created_at

5. credit_cards
- card_id VARCHAR
- account_id VARCHAR
- card_variant
- credit_limit
- available_limit
- outstanding_amt
- due_date
- min_due
- status
- issued_date
- created_at

6. card_transactions
- txn_id
- card_id VARCHAR
- txn_date
- txn_type: purchase, cashadvance, payment, refund, fee
- amount
- merchant_name
- category
- is_international
- currency
- created_at

Important date context:
- The sample banking data is for Jan 2026 to Apr 2026.
- For relative date questions like "last 3 months" or "last quarter",
  use DATE '2026-04-15' as the reference date.
"""


# ---------------------------------------------------------------------------
# LLM helper
# ---------------------------------------------------------------------------

def get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=OPENAI_CHAT_MODEL,
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0,
    )


# ---------------------------------------------------------------------------
# SQL generation
# ---------------------------------------------------------------------------

def generate_sql(question: str) -> str:
    """
    Convert natural language banking question into one safe SQL SELECT query.
    """

    llm = get_llm()

    prompt = f"""
You are a PostgreSQL expert for a Smart Banking Assistant.

Generate a single safe SQL SELECT query for the user's question.

Rules:
- Return only raw SQL.
- Do not use markdown.
- Do not use ```sql fences.
- Only SELECT statements are allowed.
- Never generate INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE, CREATE, GRANT, REVOKE.
- Use only the allowed tables and columns from the schema.
- Always add LIMIT 100 unless the query is an aggregate query.
- account_id is VARCHAR, so always compare account_id values using quotes, for example account_id = '1345367'.
- loan_id, card_id, and fd_id are VARCHAR, so always compare them using quotes.
- In transactions table, txn_type only has 'debit' or 'credit'.
- If user asks for purchase history, spend history, payments, or customer spends from transactions table, use txn_type = 'debit', not txn_type = 'purchase'.
- Use card_transactions table only for credit card transaction questions.
- For mobile numbers, return masked value only. The mobile column is already masked.
- For account transaction history, order by txn_date DESC unless user asks otherwise.
- For "last 3 months", use DATE '2026-04-15' - INTERVAL '3 months'.
- For "last quarter", use DATE_TRUNC('quarter', DATE '2026-04-15') - INTERVAL '3 months'.

Schema:
{BANKING_SCHEMA_CONTEXT}

User question:
{question}
"""

    response = llm.invoke(prompt)
    sql = response.content

    if isinstance(sql, list):
        sql = " ".join(str(item) for item in sql)

    sql = str(sql).strip()
    sql = sql.replace("```sql", "").replace("```", "").strip()

    if sql.lower().startswith("sql"):
        sql = sql[3:].strip()

    return sql


# ---------------------------------------------------------------------------
# SQL validation and safety
# ---------------------------------------------------------------------------

def validate_sql(sql: str) -> tuple[bool, str]:
    """
    Validate generated SQL before execution.

    Safety rules:
    - Must start with SELECT
    - Must not contain dangerous SQL keywords
    - Must not contain multiple statements
    - Must reference only allowed tables
    """

    if not sql or not sql.strip():
        return False, "SQL is empty."

    cleaned_sql = sql.strip()

    # Block multiple statements.
    # Allow only one optional trailing semicolon.
    if ";" in cleaned_sql.rstrip(";"):
        return False, "Multiple SQL statements are not allowed."

    cleaned_sql = cleaned_sql.rstrip(";").strip()
    lowered = cleaned_sql.lower()

    if not lowered.startswith("select"):
        return False, "Only SELECT statements are allowed."

    blocked_keywords = [
        "insert",
        "update",
        "delete",
        "drop",
        "alter",
        "truncate",
        "create",
        "grant",
        "revoke",
        "merge",
        "call",
        "execute",
    ]

    for keyword in blocked_keywords:
        if re.search(rf"\b{keyword}\b", lowered):
            return False, f"Blocked unsafe SQL keyword: {keyword}"

    allowed_tables = {
        "accounts",
        "transactions",
        "loan_accounts",
        "fixed_deposits",
        "credit_cards",
        "card_transactions",
    }

    referenced_tables = set(
        re.findall(r"\b(?:from|join)\s+([a-zA-Z_][a-zA-Z0-9_]*)", lowered)
    )

    for table in referenced_tables:
        if table not in allowed_tables:
            return False, f"Table not allowed: {table}"

    return True, cleaned_sql


def ensure_limit(sql: str, limit: int = 100) -> str:
    """
    Add LIMIT only if query does not already have LIMIT.
    """

    cleaned_sql = sql.strip().rstrip(";").strip()
    lowered = cleaned_sql.lower()

    # Already has LIMIT 100 or any LIMIT number
    if re.search(r"\blimit\s+\d+\b", lowered):
        return cleaned_sql

    aggregate_keywords = ["count(", "sum(", "avg(", "min(", "max("]

    if any(keyword in lowered for keyword in aggregate_keywords):
        return cleaned_sql

    return f"{cleaned_sql} LIMIT {limit}"


# ---------------------------------------------------------------------------
# SQL execution
# ---------------------------------------------------------------------------

def execute_sql(sql: str) -> list[dict[str, Any]]:
    """
    Execute validated SQL and return rows as list of dictionaries.
    """

    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            rows = cur.fetchall()

    return [dict(row) for row in rows]


# ---------------------------------------------------------------------------
# JSON-safe formatting
# ---------------------------------------------------------------------------

def _json_safe_value(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value)

    if isinstance(value, (date, datetime)):
        return value.isoformat()

    if isinstance(value, UUID):
        return str(value)

    return value


def make_rows_json_safe(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {key: _json_safe_value(value) for key, value in row.items()}
        for row in rows
    ]


# ---------------------------------------------------------------------------
# Response formatting
# ---------------------------------------------------------------------------

def build_simple_answer(question: str, rows: list[dict[str, Any]]) -> str:
    """
    Simple response for testing.
    Later LangGraph response_generator can make this more natural.
    """

    if not rows:
        return "No matching records found."

    if len(rows) == 1:
        return f"Found 1 matching record for: {question}"

    return f"Found {len(rows)} matching records for: {question}"


def format_sql_result(question: str, sql: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    safe_rows = make_rows_json_safe(rows)

    return {
        "query": question,
        "query_path": "sql",
        "sql_query": sql,
        "sql_result": safe_rows,
        "row_count": len(safe_rows),
        "answer": build_simple_answer(question, safe_rows),
    }


# ---------------------------------------------------------------------------
# Main NL2SQL pipeline
# ---------------------------------------------------------------------------

def ask_database(question: str) -> dict[str, Any]:
    """
    Full NL2SQL pipeline.

    Steps:
    1. Generate SQL from natural language
    2. Validate SQL safety
    3. Add LIMIT if needed
    4. Execute SQL
    5. Return structured result
    """

    print(f"[ask_database] Question: {question}")

    generated_sql = generate_sql(question)
    print(f"[ask_database] Generated SQL:\n{generated_sql}")

    is_valid, validation_result = validate_sql(generated_sql)

    if not is_valid:
        return {
            "query": question,
            "query_path": "sql",
            "status": "blocked",
            "sql_query": generated_sql,
            "sql_result": [],
            "row_count": 0,
            "answer": f"SQL blocked for safety reason: {validation_result}",
        }

    safe_sql = ensure_limit(validation_result)
    print(f"[ask_database] Safe SQL:\n{safe_sql}")

    try:
        rows = execute_sql(safe_sql)
    except Exception as exc:
        return {
            "query": question,
            "query_path": "sql",
            "status": "error",
            "sql_query": safe_sql,
            "sql_result": [],
            "row_count": 0,
            "answer": f"SQL execution error: {exc}",
        }

    response = format_sql_result(
        question=question,
        sql=safe_sql,
        rows=rows,
    )

    response["status"] = "success"
    return response


# ---------------------------------------------------------------------------
# Local testing
# ---------------------------------------------------------------------------

def print_response(response: dict[str, Any]) -> None:
    print("\nNL2SQL Response")
    print("-" * 80)
    print(f"Status    : {response.get('status')}")
    print(f"Path      : {response.get('query_path')}")
    print(f"Answer    : {response.get('answer')}")
    print(f"Row Count : {response.get('row_count')}")

    print("\nSQL:")
    print(response.get("sql_query"))

    print("\nRows:")
    rows = response.get("sql_result", [])

    for index, row in enumerate(rows[:10], start=1):
        print(f"{index}. {row}")


if __name__ == "__main__":
    test_questions = [
        "Give me the last 3 months purchase history of customer account 1345367",
        "What is the current outstanding balance and next EMI due date for loan account L-789012?",
        "List all transactions above 50000 for account 1345367",
        "Show me all active FDs for account 1345367",
        "Show international transactions on credit card CC-881001",
    ]

    for question in test_questions:
        response = ask_database(question)
        print_response(response)
        print("\n" + "=" * 100 + "\n")