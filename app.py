import os
import io
import json
import math
import uuid
import shutil
import sqlite3
import hashlib
import tempfile
from datetime import datetime, timedelta, timezone, date
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    import cv2
except Exception:
    cv2 = None


st.set_page_config(
    page_title="Invoice Financing & Receivables Risk Intelligence",
    page_icon="🧾",
    layout="wide",
    initial_sidebar_state="expanded",
)


DB_PATH = os.getenv("IFRRI_DB_PATH", "invoice_financing_demo.db")
UPLOAD_DIR = "uploads"
SCHEMA_VERSION = "1.0.0"
RULES_VERSION = "rules-1.0.0"
MAX_AMOUNT = 10_000_000.0
ALLOWED_CURRENCIES = {
    "USD", "EUR", "GBP", "JPY", "INR", "BRL", "MXN", "AUD", "CAD", "CHF",
    "SGD", "AED", "ZAR", "TRY", "IDR", "PHP", "VND", "THB", "MYR", "KRW"
}
BUCKET_ORDER = ["Current", "1-30", "31-60", "61-90", "90+"]

ROLES = ["Administrator", "Analyst/Operator", "Reviewer", "Read-only Stakeholder"]
PERMISSIONS = {
    "Administrator": {
        "view", "import", "calculate", "prepare_review", "review", "approve",
        "override", "export", "settings", "audit", "backup", "restore", "purge"
    },
    "Analyst/Operator": {
        "view", "import", "calculate", "prepare_review", "export"
    },
    "Reviewer": {
        "view", "calculate", "review", "approve", "override", "export"
    },
    "Read-only Stakeholder": {
        "view", "export"
    },
}

HARD_FLAGS = {
    "missing_debtor",
    "duplicate_invoice",
    "missing_issue_date",
    "missing_due_date",
    "future_issue_date",
    "due_before_issue",
    "missing_or_nonpositive_amount",
    "amount_out_of_range",
    "unsupported_currency",
    "contradictory_payment",
}


def uid() -> str:
    return str(uuid.uuid4())


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def ensure_dirs() -> None:
    os.makedirs(UPLOAD_DIR, exist_ok=True)


def get_conn() -> sqlite3.Connection:
    ensure_dirs()
    conn = sqlite3.connect(DB_PATH, timeout=15)
    conn.row_factory = sqlite3.Row
    return conn


def safe_filename(name: str) -> str:
    name = os.path.basename(name)
    name = "".join(ch for ch in name if ch.isalnum() or ch in "._-")
    return (name[:120] or "upload.bin")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def safe_float(v: Any, default: float = math.nan) -> float:
    try:
        if v is None:
            return default
        if isinstance(v, (pd.Timestamp, datetime, date)):
            return default
        if pd.isna(v):
            return default
        return float(v)
    except Exception:
        return default


def to_ts(v: Any) -> pd.Timestamp:
    try:
        if v is None:
            return pd.NaT
        if isinstance(v, pd.Timestamp):
            return v
        return pd.to_datetime(v, errors="coerce")
    except Exception:
        return pd.NaT


def to_date(v: Any) -> Optional[date]:
    ts = to_ts(v)
    if pd.isna(ts):
        return None
    return ts.date()


def get_val(row: Any, col: str) -> str:
    try:
        if col not in row.index:
            return ""
        v = row[col]
        if v is None:
            return ""
        if isinstance(v, float) and math.isnan(v):
            return ""
        if pd.isna(v):
            return ""
        return str(v).strip()
    except Exception:
        return ""


def json_ready(obj: Any) -> Any:
    return json.loads(json.dumps(obj, default=str))


def has_perm(role: str, perm: str) -> bool:
    return perm in PERMISSIONS.get(role, set())


def init_schema() -> None:
    with get_conn() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at TEXT
            );

            CREATE TABLE IF NOT EXISTS audit_events (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                actor TEXT NOT NULL,
                role TEXT NOT NULL,
                action TEXT NOT NULL,
                entity TEXT NOT NULL,
                entity_id TEXT,
                details TEXT
            );

            CREATE TABLE IF NOT EXISTS organizations (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                country TEXT,
                currency TEXT,
                sector TEXT,
                credit_limit REAL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                source TEXT,
                schema_version TEXT
            );

            CREATE TABLE IF NOT EXISTS invoices (
                id TEXT PRIMARY KEY,
                invoice_number TEXT,
                debtor_id TEXT,
                debtor_name TEXT,
                issue_date TEXT,
                due_date TEXT,
                amount REAL,
                currency TEXT,
                description TEXT,
                lifecycle_status TEXT DEFAULT 'draft',
                validity_score REAL,
                risk_score REAL,
                risk_flags TEXT,
                recommended_decision TEXT,
                final_decision TEXT,
                reviewed_by TEXT,
                reviewed_at TEXT,
                override_reason TEXT,
                source TEXT,
                source_file_id TEXT,
                row_number INTEGER,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                schema_version TEXT
            );

            CREATE TABLE IF NOT EXISTS payments (
                id TEXT PRIMARY KEY,
                invoice_id TEXT,
                invoice_number TEXT,
                debtor_id TEXT,
                payment_date TEXT,
                amount REAL,
                currency TEXT,
                reference TEXT,
                source TEXT,
                source_file_id TEXT,
                row_number INTEGER,
                created_at TEXT NOT NULL,
                schema_version TEXT
            );

            CREATE TABLE IF NOT EXISTS decisions (
                id TEXT PRIMARY KEY,
                entity_type TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                decision TEXT NOT NULL,
                requested_amount REAL,
                approved_amount REAL,
                risk_score REAL,
                rationale TEXT,
                override_reason TEXT,
                actor TEXT NOT NULL,
                role TEXT NOT NULL,
                created_at TEXT NOT NULL,
                schema_version TEXT
            );

            CREATE TABLE IF NOT EXISTS uploads (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                stored_path TEXT,
                checksum TEXT,
                size_bytes INTEGER,
                uploaded_at TEXT NOT NULL,
                actor TEXT,
                role TEXT,
                entity_type TEXT,
                status TEXT,
                rows_total INTEGER,
                rows_valid INTEGER,
                error_report_path TEXT,
                schema_version TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_invoices_debtor ON invoices(debtor_id);
            CREATE INDEX IF NOT EXISTS idx_invoices_status ON invoices(lifecycle_status);
            CREATE INDEX IF NOT EXISTS idx_payments_invoice ON payments(invoice_id);
            CREATE INDEX IF NOT EXISTS idx_audit_created ON audit_events(created_at);
            """
        )
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value, updated_at) VALUES (?, ?, ?)",
            ("schema_version", SCHEMA_VERSION, utcnow().isoformat()),
        )
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value, updated_at) VALUES (?, ?, ?)",
            ("rules_version", RULES_VERSION, utcnow().isoformat()),
        )
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value, updated_at) VALUES (?, ?, ?)",
            ("synthetic_demo", "true", utcnow().isoformat()),
        )
        conn.commit()


def audit(conn: sqlite3.Connection, actor: str, role: str, action: str,
          entity: str, entity_id: str, details: Any = "") -> None:
    conn.execute(
        """
        INSERT INTO audit_events
            (id, created_at, actor, role, action, entity, entity_id, details)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            uid(),
            utcnow().isoformat(),
            actor,
            role,
            action,
            entity,
            entity_id,
            json.dumps(details, default=str),
        ),
    )


def save_original_file(data: bytes, filename: str, entity_type: str,
                       actor: str, role: str, status: str) -> Tuple[str, str, str]:
    ensure_dirs()
    upload_id = uid()
    stored = os.path.join(UPLOAD_DIR, f"{upload_id}_{safe_filename(filename)}")
    with open(stored, "wb") as f:
        f.write(data)
    checksum = sha256_bytes(data)
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO uploads
                (id, filename, stored_path, checksum, size_bytes, uploaded_at,
                 actor, role, entity_type, status, rows_total, rows_valid,
                 error_report_path, schema_version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                upload_id,
                filename,
                stored,
                checksum,
                len(data),
                utcnow().isoformat(),
                actor,
                role,
                entity_type,
                status,
                0,
                0,
                None,
                SCHEMA_VERSION,
            ),
        )
        audit(conn, actor, role, "upload_saved", "upload", upload_id,
              {"filename": filename, "entity_type": entity_type, "status": status})
        conn.commit()
    return upload_id, stored, checksum


def save_error_report(errors_df: pd.DataFrame, upload_id: str) -> str:
    ensure_dirs()
    path = os.path.join(UPLOAD_DIR, f"error_{upload_id}.csv")
    errors_df.to_csv(path, index=False)
    return path


def mark_upload(upload_id: str, status: str, rows_total: int,
                rows_valid: int, error_report_path: Optional[str] = None) -> None:
    with get_conn() as conn:
        conn.execute(
            """
            UPDATE uploads
            SET status = ?, rows_total = ?, rows_valid = ?, error_report_path = ?
            WHERE id = ?
            """,
            (status, rows_total, rows_valid, error_report_path, upload_id),
        )
        conn.commit()


def seed_if_empty() -> None:
    with get_conn() as conn:
        count = conn.execute("SELECT COUNT(*) AS c FROM invoices").fetchone()["c"]
        if count > 0:
            return

        rng = np.random.default_rng(20261112)
        today = utcnow().date()

        debtors = [
            ("D-1001", "Northwind Retail", "US", "USD", "Retail", 750000, 0.16),
            ("D-1002", "Helios Manufacturing", "DE", "EUR", "Industrial", 900000, 0.14),
            ("D-1003", "Blue Harbor Logistics", "SG", "SGD", "Logistics", 500000, 0.12),
            ("D-1004", "Andes Foods", "BR", "BRL", "Food", 400000, 0.10),
            ("D-1005", "Sakura Electronics", "JP", "JPY", "Electronics", 850000, 0.10),
            ("D-1006", "Atlas Mining Services", "ZA", "ZAR", "Mining", 600000, 0.09),
            ("D-1007", "Meridian Health Supplies", "UK", "GBP", "Healthcare", 450000, 0.08),
            ("D-1008", "Qantas Regional Freight", "AU", "AUD", "Transport", 300000, 0.06),
            ("D-1009", "Iberia Textiles", "ES", "EUR", "Textiles", 250000, 0.05),
            ("D-1010", "Maple Leaf Components", "CA", "CAD", "Manufacturing", 350000, 0.04),
            ("D-1011", "Dubai Bay Trading", "AE", "AED", "Trading", 300000, 0.03),
            ("D-1012", "Seoul Precision Parts", "KR", "KRW", "Precision", 250000, 0.03),
        ]

        for d in debtors:
            conn.execute(
                """
                INSERT INTO organizations
                    (id, name, country, currency, sector, credit_limit,
                     created_at, updated_at, source, schema_version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (d[0], d[1], d[2], d[3], d[4], float(d[5]),
                 utcnow().isoformat(), utcnow().isoformat(), "synthetic-demo", SCHEMA_VERSION),
            )

        weights = np.array([d[6] for d in debtors], dtype=float)
        weights = weights / weights.sum()
        debtor_ids = [d[0] for d in debtors]
        debtor_currency = {d[0]: d[3] for d in debtors}
        debtor_name = {d[0]: d[1] for d in debtors}

        descriptions = [
            "Software subscription", "Logistics services", "Components supply",
            "Consulting fees", "Equipment lease", "Marketing services",
            "Warehouse handling", "Maintenance contract"
        ]

        def insert_invoice(inv_id: str, invoice_number: str, debtor_id: str,
                           issue: date, due: date, amount: float, currency: str,
                           description: str, lifecycle: str = "validated",
                           source: str = "synthetic-demo") -> None:
            conn.execute(
                """
                INSERT INTO invoices
                    (id, invoice_number, debtor_id, debtor_name, issue_date, due_date,
                     amount, currency, description, lifecycle_status, source,
                     created_at, updated_at, schema_version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    inv_id, invoice_number, debtor_id, debtor_name.get(debtor_id, "Unknown"),
                    issue.isoformat(), due.isoformat(), float(amount), currency,
                    description, lifecycle, source, utcnow().isoformat(),
                    utcnow().isoformat(), SCHEMA_VERSION,
                ),
            )

        def insert_payment(pay_id: str, invoice_id: str, invoice_number: str,
                           debtor_id: str, pay_date: date, amount: float,
                           currency: str, reference: str, source: str = "synthetic-demo") -> None:
            conn.execute(
                """
                INSERT INTO payments
                    (id, invoice_id, invoice_number, debtor_id, payment_date,
                     amount, currency, reference, source, created_at, schema_version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    pay_id, invoice_id, invoice_number, debtor_id, pay_date.isoformat(),
                    float(amount), currency, reference, source, utcnow().isoformat(),
                    SCHEMA_VERSION,
                ),
            )

        for n in range(1, 141):
            did = str(rng.choice(debtor_ids, p=weights))
            issue_offset = int(rng.integers(1, 181))
            issue = today - timedelta(days=issue_offset)
            terms = int(rng.choice([15, 30, 45, 60, 75]))
            due = issue + timedelta(days=terms)
            amount = round(float(np.exp(rng.normal(10.0, 0.9))), 2)
            if not np.isfinite(amount) or amount <= 0 or amount > MAX_AMOUNT:
                amount = round(float(rng.uniform(1000, 250000)), 2)
            currency = debtor_currency[did]
            inv_id = uid()
            inv_no = f"DEMO-{issue:%y%m%d}-{n:04d}"
            desc = str(rng.choice(descriptions))
            insert_invoice(inv_id, inv_no, did, issue, due, amount, currency, desc)

            if due < today:
                slow_payer = did in {"D-1008", "D-1009"}
                pay_prob = 0.58 if slow_payer else 0.84
                if float(rng.random()) < pay_prob:
                    delay_mean = 12 if slow_payer else 2
                    delay = int(rng.normal(delay_mean, 10))
                    delay = max(-14, min(90, delay))
                    pay_date = due + timedelta(days=delay)
                    if pay_date > today:
                        pay_date = today - timedelta(days=int(rng.integers(0, 3)))
                    pay_amt = amount
                    if float(rng.random()) < 0.20:
                        pay_amt = round(amount * float(rng.uniform(0.45, 0.90)), 2)
                    insert_payment(uid(), inv_id, inv_no, did, pay_date, pay_amt,
                                   currency, f"PAY-{inv_no}-1")
                    if pay_amt < amount and float(rng.random()) < 0.65:
                        rem = round(amount - pay_amt, 2)
                        pay2_date = min(today, pay_date + timedelta(days=int(rng.integers(5, 35))))
                        insert_payment(uid(), inv_id, inv_no, did, pay2_date, rem,
                                       currency, f"PAY-{inv_no}-2")

        # Deliberate exception cases for workflow testing.
        dup_id_1 = uid()
        dup_id_2 = uid()
        dup_issue = today - timedelta(days=20)
        dup_due = today + timedelta(days=10)
        insert_invoice(dup_id_1, "DEMO-DUP-0001", "D-1001", dup_issue, dup_due,
                       15000.0, "USD", "Duplicate invoice number test", "exception")
        insert_invoice(dup_id_2, "DEMO-DUP-0001", "D-1001", dup_issue, dup_due,
                       15000.0, "USD", "Duplicate invoice number test", "exception")

        insert_invoice(uid(), "DEMO-UNKNOWN-0001", "D-9999",
                       today - timedelta(days=18), today + timedelta(days=12),
                       9500.0, "USD", "Unknown debtor test", "exception")

        insert_invoice(uid(), "DEMO-FUTURE-0001", "D-1002",
                       today + timedelta(days=10), today + timedelta(days=40),
                       12000.0, "EUR", "Future issue date test", "exception")

        insert_invoice(uid(), "DEMO-NEGATIVE-0001", "D-1003",
                       today - timedelta(days=25), today - timedelta(days=5),
                       -2500.0, "SGD", "Negative amount test", "exception")

        insert_invoice(uid(), "DEMO-DUE-BEFORE-0001", "D-1004",
                       today - timedelta(days=30), today - timedelta(days=45),
                       7000.0, "BRL", "Due before issue test", "exception")

        insert_invoice(uid(), "DEMO-HUGE-0001", "D-1005",
                       today - timedelta(days=12), today + timedelta(days=48),
                       25000000.0, "JPY", "Out-of-range amount test", "exception")

        insert_invoice(uid(), "DEMO-CURRENCY-0001", "D-1006",
                       today - timedelta(days=16), today + timedelta(days=14),
                       8000.0, "XXX", "Unsupported currency test", "exception")

        contra_id = uid()
        contra_issue = today - timedelta(days=40)
        contra_due = today - timedelta(days=10)
        insert_invoice(contra_id, "DEMO-CONTRA-0001", "D-1007", contra_issue, contra_due,
                       10000.0, "GBP", "Contradictory payment test", "exception")
        insert_payment(uid(), contra_id, "DEMO-CONTRA-0001", "D-1007",
                       today - timedelta(days=50), 12000.0, "GBP", "PAY-CONTRA-OVER")

        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value, updated_at) VALUES (?, ?, ?)",
            ("seeded_at", utcnow().isoformat(), utcnow().isoformat()),
        )
        audit(conn, "system", "Administrator", "seed", "database", "demo",
              "Synthetic demo data loaded. No real customer data.")
        conn.commit()

    refresh_stored_scores(today)


def compute_invoice_assessment(inv: Dict[str, Any], debtor: Optional[Dict[str, Any]],
                               payments: List[Dict[str, Any]], as_of_ts: pd.Timestamp,
                               duplicate_count: int) -> Dict[str, Any]:
    flags: List[str] = []
    deductions: List[Dict[str, Any]] = []
    warnings: List[str] = []

    def deduct(flag: str, impact: float, message: str) -> None:
        flags.append(flag)
        deductions.append({"flag": flag, "impact": float(impact), "message": message})

    amount = safe_float(inv.get("amount"))
    issue_ts = to_ts(inv.get("issue_date"))
    due_ts = to_ts(inv.get("due_date"))
    currency = str(inv.get("currency") or "").strip().upper()
    description = str(inv.get("description") or "").strip()
    debtor_id = str(inv.get("debtor_id") or "").strip()

    paid_sum = 0.0
    contradictory_payment = False

    for p in payments:
        p_ts = to_ts(p.get("payment_date"))
        if pd.isna(p_ts) or p_ts > as_of_ts:
            continue
        p_amt = safe_float(p.get("amount"), 0.0)
        if not np.isfinite(p_amt):
            p_amt = 0.0
        paid_sum += p_amt
        if not pd.isna(issue_ts) and p_ts < issue_ts:
            contradictory_payment = True

    if not debtor or not debtor_id:
        deduct("missing_debtor", 40, "Debtor reference is missing or unknown.")

    if duplicate_count > 1:
        deduct("duplicate_invoice", 30, "Duplicate invoice number for the same debtor.")

    if pd.isna(issue_ts):
        deduct("missing_issue_date", 20, "Issue date is missing or unparseable.")
    elif issue_ts > as_of_ts:
        deduct("future_issue_date", 25, "Issue date is after the as-of date.")

    if pd.isna(due_ts):
        deduct("missing_due_date", 20, "Due date is missing or unparseable.")
    elif not pd.isna(issue_ts) and due_ts < issue_ts:
        deduct("due_before_issue", 20, "Due date is before issue date.")

    if not np.isfinite(amount) or amount <= 0:
        deduct("missing_or_nonpositive_amount", 40, "Amount is missing, zero, or negative.")
    elif amount > MAX_AMOUNT:
        deduct("amount_out_of_range", 15, "Amount exceeds configured maximum.")

    if currency not in ALLOWED_CURRENCIES:
        deduct("unsupported_currency", 10, "Currency is missing or not in reference catalog.")

    if np.isfinite(amount) and amount > 0 and paid_sum > amount * 1.02:
        contradictory_payment = True

    if contradictory_payment:
        deduct("contradictory_payment", 25, "Payment data contradicts invoice amount or dates.")

    outstanding = 0.0
    if np.isfinite(amount) and amount > 0:
        outstanding = max(0.0, amount - paid_sum)

    days_past_due = 0
    if not pd.isna(due_ts) and outstanding > 0:
        days_past_due = max(0, int((as_of_ts - due_ts).days))
        if days_past_due > 120:
            deduct("stale_unpaid", 10, "Invoice remains unpaid more than 120 days past due.")

    if not description:
        warnings.append("Missing invoice description.")
    if not pd.isna(due_ts) and due_ts > as_of_ts + timedelta(days=365):
        warnings.append("Due date is more than one year beyond as-of date.")
    if np.isfinite(amount) and amount > 1_000_000:
        warnings.append("Large invoice amount; verify counterparty and goods/service evidence.")

    total_deduction = sum(float(d["impact"]) for d in deductions)
    validity_score = max(0.0, 100.0 - total_deduction)

    if days_past_due <= 0:
        aging_bucket = "Current"
    elif days_past_due <= 30:
        aging_bucket = "1-30"
    elif days_past_due <= 60:
        aging_bucket = "31-60"
    elif days_past_due <= 90:
        aging_bucket = "61-90"
    else:
        aging_bucket = "90+"

    validity_risk = 100.0 - validity_score
    aging_risk = 0.0
    if outstanding > 0:
        aging_risk = min(100.0, max(0.0, days_past_due / 120.0 * 100.0))

    risk_score = round(0.6 * validity_risk + 0.4 * aging_risk, 1)

    hard_present = bool(set(flags) & HARD_FLAGS)
    if hard_present:
        recommended = "Reject"
    elif risk_score <= 45 and validity_score >= 80:
        recommended = "Approve"
    elif risk_score <= 70:
        recommended = "Review"
    else:
        recommended = "Reject"

    contributors = list(deductions)
    if aging_risk > 0:
        contributors.append({
            "flag": "aging_risk",
            "impact": round(0.4 * aging_risk, 1),
            "message": f"Outstanding amount is {days_past_due} days past due.",
        })
    contributors.sort(key=lambda x: abs(float(x.get("impact", 0))), reverse=True)

    return {
        "validity_score": round(validity_score, 1),
        "risk_score": risk_score,
        "flags": sorted(set(flags)),
        "hard_flags": sorted(set(flags) & HARD_FLAGS),
        "deductions": deductions,
        "contributors": contributors,
        "warnings": warnings,
        "paid_amount": round(paid_sum, 2),
        "outstanding": round(outstanding, 2),
        "days_past_due": days_past_due,
        "aging_bucket": aging_bucket,
        "recommended_decision": recommended,
        "rules_version": RULES_VERSION,
    }


def calculate_concentration(exposure: pd.Series, total_outstanding: float,
                            debtor_map: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    empty = {
        "hhi": 0.0,
        "hhi_10k": 0.0,
        "top1_share": 0.0,
        "top3_share": 0.0,
        "risk_score": 0.0,
        "top_debtors": [],
    }
    if total_outstanding <= 0 or exposure is None or exposure.empty:
        return empty

    exposure = exposure[exposure > 0].sort_values(ascending=False)
    shares = exposure / float(total_outstanding)
    hhi = float((shares ** 2).sum())
    top1 = float(shares.iloc[0]) if len(shares) > 0 else 0.0
    top3 = float(shares.head(3).sum())
    risk = min(100.0, max(0.0, (top1 * 0.7 + top3 * 0.3) * 100.0))

    top_debtors = []
    for did, val in exposure.head(5).items():
        top_debtors.append({
            "debtor_id": str(did),
            "name": str(debtor_map.get(did, {}).get("name", "Unknown")),
            "exposure": float(val),
            "share": float(shares.get(did, 0.0)),
        })

    return {
        "hhi": round(hhi, 4),
        "hhi_10k": round(hhi * 10000.0, 1),
        "top1_share": round(top1, 4),
        "top3_share": round(top3, 4),
        "risk_score": round(risk, 1),
        "top_debtors": top_debtors,
    }


def empty_metrics(as_of: pd.Timestamp) -> Dict[str, Any]:
    return {
        "as_of": as_of.date().isoformat(),
        "total_invoices": 0,
        "total_amount": 0.0,
        "total_outstanding": 0.0,
        "overdue_amount": 0.0,
        "overdue_pct": 0.0,
        "avg_validity": 100.0,
        "validity_risk": 0.0,
        "aging_risk": 0.0,
        "concentration_risk": 0.0,
        "delay_risk": 0.0,
        "portfolio_risk": 0.0,
        "weighted_days_past_due": 0.0,
        "severe_share": 0.0,
        "avg_delay_days": 0.0,
        "median_delay_days": 0.0,
        "pct_late": 0.0,
        "hhi": 0.0,
        "hhi_10k": 0.0,
        "top1_share": 0.0,
        "top3_share": 0.0,
        "top_debtors": [],
        "aging_buckets": {b: 0.0 for b in BUCKET_ORDER},
        "rules_version": RULES_VERSION,
        "schema_version": SCHEMA_VERSION,
        "synthetic": True,
        "human_approval_required": True,
    }


def calculate_portfolio_metrics(as_of: Any) -> Tuple[Dict[str, Any], pd.DataFrame]:
    as_of_ts = pd.Timestamp(as_of)

    with get_conn() as conn:
        invoices = pd.read_sql_query("SELECT * FROM invoices", conn)
        payments = pd.read_sql_query("SELECT * FROM payments", conn)
        debtors = pd.read_sql_query("SELECT * FROM organizations", conn)

    if invoices.empty:
        return empty_metrics(as_of_ts), pd.DataFrame()

    invoices["amount"] = pd.to_numeric(invoices["amount"], errors="coerce")
    invoices["issue_date"] = pd.to_datetime(invoices["issue_date"], errors="coerce")
    invoices["due_date"] = pd.to_datetime(invoices["due_date"], errors="coerce")

    if not payments.empty:
        payments["payment_date"] = pd.to_datetime(payments["payment_date"], errors="coerce")
        payments["amount"] = pd.to_numeric(payments["amount"], errors="coerce")
        payments = payments[payments["payment_date"] <= as_of_ts]
    else:
        payments = pd.DataFrame(columns=["id", "invoice_id", "payment_date", "amount"])

    paid_by_inv = payments.groupby("invoice_id")["amount"].sum() if not payments.empty else pd.Series(dtype=float)
    final_pay_by_inv = payments.groupby("invoice_id")["payment_date"].max() if not payments.empty else pd.Series(dtype="datetime64[ns]")

    invoices["paid_amount_mapped"] = invoices["id"].map(paid_by_inv).fillna(0.0)

    debtor_map = debtors.set_index("id").to_dict("index") if not debtors.empty else {}
    payments_by_inv = {k: g.to_dict("records") for k, g in payments.groupby("invoice_id")} if not payments.empty else {}

    dup_counts = invoices.groupby(["invoice_number", "debtor_id"], dropna=False)["id"].transform("count")
    invoices["duplicate_count"] = dup_counts.fillna(1)

    rows: List[Dict[str, Any]] = []
    for _, inv in invoices.iterrows():
        deb = debtor_map.get(inv["debtor_id"])
        pays = payments_by_inv.get(inv["id"], [])
        dup = int(inv["duplicate_count"]) if pd.notna(inv["duplicate_count"]) else 1
        ass = compute_invoice_assessment(inv.to_dict(), deb, pays, as_of_ts, dup)
        rows.append({
            "id": inv["id"],
            "invoice_number": inv["invoice_number"],
            "debtor_id": inv["debtor_id"],
            "debtor_name": (deb.get("name") if deb else inv.get("debtor_name")) or "Unknown",
            "issue_date": inv["issue_date"],
            "due_date": inv["due_date"],
            "amount": safe_float(inv["amount"]),
            "currency": inv["currency"],
            "description": inv.get("description", ""),
            "paid_amount": ass["paid_amount"],
            "outstanding": ass["outstanding"],
            "days_past_due": ass["days_past_due"],
            "aging_bucket": ass["aging_bucket"],
            "validity_score": ass["validity_score"],
            "risk_score": ass["risk_score"],
            "flags": ass["flags"],
            "hard_flags": ass["hard_flags"],
            "contributors": ass["contributors"],
            "warnings": ass["warnings"],
            "recommended_decision": ass["recommended_decision"],
            "lifecycle_status": inv.get("lifecycle_status", "draft"),
            "final_decision": inv.get("final_decision", ""),
            "reviewed_by": inv.get("reviewed_by", ""),
            "reviewed_at": inv.get("reviewed_at", ""),
            "credit_limit": safe_float(deb.get("credit_limit")) if deb else math.nan,
            "rules_version": ass["rules_version"],
        })

    scored = pd.DataFrame(rows)
    active = scored[scored["issue_date"] <= as_of_ts].copy()

    if active.empty:
        metrics = empty_metrics(as_of_ts)
        return metrics, scored

    total_outstanding = float(active["outstanding"].sum())
    overdue_amount = float(active[active["days_past_due"] > 0]["outstanding"].sum())
    overdue_pct = (overdue_amount / total_outstanding * 100.0) if total_outstanding > 0 else 0.0

    bucket_sums = (
        active.groupby("aging_bucket")["outstanding"]
        .sum()
        .reindex(BUCKET_ORDER)
        .fillna(0.0)
    )

    exposure = active.groupby("debtor_id")["outstanding"].sum()
    concentration = calculate_concentration(exposure, total_outstanding, debtor_map)

    paid_full = active[(active["amount"] > 0) & (active["paid_amount"] >= active["amount"] * 0.99)].copy()
    avg_delay = 0.0
    median_delay = 0.0
    pct_late = 0.0
    if not paid_full.empty:
        paid_full["final_payment_date"] = paid_full["id"].map(final_pay_by_inv)
        paid_full = paid_full.dropna(subset=["final_payment_date", "due_date"])
        if not paid_full.empty:
            delays = (paid_full["final_payment_date"] - paid_full["due_date"]).dt.days
            avg_delay = float(delays.mean()) if len(delays) else 0.0
            median_delay = float(delays.median()) if len(delays) else 0.0
            pct_late = float((delays > 0).mean()) if len(delays) else 0.0

    avg_validity = float(active["validity_score"].mean())
    validity_risk = max(0.0, 100.0 - avg_validity)

    weighted_days = 0.0
    severe_share = 0.0
    if total_outstanding > 0:
        weighted_days = float((active["outstanding"] * active["days_past_due"]).sum() / total_outstanding)
        severe_share = float(bucket_sums.get("90+", 0.0) / total_outstanding)

    aging_risk = min(100.0, max(0.0, 0.7 * (weighted_days / 90.0 * 100.0) + 0.3 * (severe_share * 100.0)))
    delay_risk = min(100.0, max(0.0, 0.7 * (max(0.0, avg_delay) / 60.0 * 100.0) + 0.3 * (pct_late * 100.0)))
    concentration_risk = float(concentration["risk_score"])

    portfolio_risk = round(
        0.35 * validity_risk
        + 0.25 * aging_risk
        + 0.20 * concentration_risk
        + 0.20 * delay_risk,
        1,
    )

    metrics = {
        "as_of": as_of_ts.date().isoformat(),
        "total_invoices": int(len(active)),
        "total_amount": float(active["amount"].sum(skipna=True)),
        "total_outstanding": round(total_outstanding, 2),
        "overdue_amount": round(overdue_amount, 2),
        "overdue_pct": round(overdue_pct, 1),
        "avg_validity": round(avg_validity, 1),
        "validity_risk": round(validity_risk, 1),
        "aging_risk": round(aging_risk, 1),
        "concentration_risk": round(concentration_risk, 1),
        "delay_risk": round(delay_risk, 1),
        "portfolio_risk": portfolio_risk,
        "weighted_days_past_due": round(weighted_days, 1),
        "severe_share": round(severe_share, 4),
        "avg_delay_days": round(avg_delay, 1),
        "median_delay_days": round(median_delay, 1),
        "pct_late": round(pct_late, 4),
        "hhi": concentration["hhi"],
        "hhi_10k": concentration["hhi_10k"],
        "top1_share": concentration["top1_share"],
        "top3_share": concentration["top3_share"],
        "top_debtors": concentration["top_debtors"],
        "aging_buckets": {k: float(v) for k, v in bucket_sums.items()},
        "rules_version": RULES_VERSION,
        "schema_version": SCHEMA_VERSION,
        "synthetic": True,
        "human_approval_required": True,
    }

    return metrics, scored


def refresh_stored_scores(as_of: Any) -> Dict[str, Any]:
    metrics, scored = calculate_portfolio_metrics(as_of)
    if scored.empty:
        return metrics

    with get_conn() as conn:
        for _, row in scored.iterrows():
            conn.execute(
                """
                UPDATE invoices
                SET validity_score = ?, risk_score = ?, risk_flags = ?,
                    recommended_decision = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    float(row["validity_score"]),
                    float(row["risk_score"]),
                    json.dumps(row["flags"], default=str),
                    str(row["recommended_decision"]),
                    utcnow().isoformat(),
                    str(row["id"]),
                ),
            )
        audit(conn, "system", "Administrator", "refresh_scores", "portfolio",
              str(pd.Timestamp(as_of).date()), {"rows": int(len(scored))})
        conn.commit()

    return metrics


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [
        str(c).strip().lower().replace(" ", "_").replace("-", "_")
        for c in df.columns
    ]
    return df


def add_error(errors: List[Dict[str, Any]], row: int, field: str, issue: str,
              value: Any, severity: str = "error", suggestion: str = "") -> None:
    errors.append({
        "row": int(row),
        "field": field,
        "issue": issue,
        "value": "" if value is None else str(value),
        "severity": severity,
        "suggestion": suggestion,
    })


def validate_import(raw_df: pd.DataFrame, entity_type: str,
                    conn: sqlite3.Connection) -> Tuple[pd.DataFrame, pd.DataFrame]:
    errors: List[Dict[str, Any]] = []
    df = normalize_columns(raw_df)
    today = utcnow().date()
    today_ts = pd.Timestamp(today)

    error_columns = ["row", "field", "issue", "value", "severity", "suggestion"]

    if df.empty:
        add_error(errors, 0, "file", "Empty file", "", "error", "Upload a file with at least one data row.")
        return df, pd.DataFrame(errors, columns=error_columns)

    if entity_type == "invoices":
        required = ["invoice_number", "issue_date", "due_date", "amount", "currency"]
        missing = [c for c in required if c not in df.columns]
        if "debtor_id" not in df.columns and "debtor_name" not in df.columns:
            missing.append("debtor_id or debtor_name")
        if missing:
            for m in missing:
                add_error(errors, 0, m, "Required column missing", "", "error", f"Add column: {m}")
            return df, pd.DataFrame(errors, columns=error_columns)

        df["row_number"] = df.index + 2

        debtor_rows = pd.read_sql_query("SELECT id, name FROM organizations", conn)
        debtor_id_set = set(debtor_rows["id"].astype(str)) if not debtor_rows.empty else set()
        name_to_id = {
            str(n).strip().lower(): str(i)
            for i, n in zip(debtor_rows["id"], debtor_rows["name"])
        } if not debtor_rows.empty else {}

        resolved_debtor_ids: List[str] = []
        for _, row in df.iterrows():
            rn = int(row["row_number"])
            did = get_val(row, "debtor_id")
            dname = get_val(row, "debtor_name")
            if did and did not in debtor_id_set:
                add_error(errors, rn, "debtor_id", "Unknown debtor id", did, "error",
                          "Import debtor first or correct debtor_id.")
            elif not did and dname:
                did = name_to_id.get(dname.lower(), "")
                if not did:
                    add_error(errors, rn, "debtor_name", "Unknown debtor name", dname, "error",
                              "Import debtor first or correct debtor_name.")
            elif not did and not dname:
                add_error(errors, rn, "debtor_id", "Missing debtor reference", "", "error",
                          "Provide debtor_id or debtor_name.")
            resolved_debtor_ids.append(did)

        df["debtor_id_resolved"] = resolved_debtor_ids
        df["issue_date_dt"] = pd.to_datetime(df["issue_date"], errors="coerce")
        df["due_date_dt"] = pd.to_datetime(df["due_date"], errors="coerce")
        df["amount_num"] = pd.to_numeric(df["amount"], errors="coerce")
        df["currency_clean"] = df["currency"].astype(str).str.upper().str.strip()

        for _, row in df.iterrows():
            rn = int(row["row_number"])
            issue = row["issue_date_dt"]
            due = row["due_date_dt"]
            amount = row["amount_num"]
            currency = row["currency_clean"]
            inv_no = get_val(row, "invoice_number")

            if not inv_no:
                add_error(errors, rn, "invoice_number", "Missing invoice number", "", "error",
                          "Provide a non-empty invoice number.")
            if pd.isna(issue):
                add_error(errors, rn, "issue_date", "Unparseable issue date", row["issue_date"], "error",
                          "Use ISO date YYYY-MM-DD.")
            elif issue.date() > today:
                add_error(errors, rn, "issue_date", "Future issue date", issue.date().isoformat(), "error",
                          "Issue date cannot be after today for imported invoices.")
            if pd.isna(due):
                add_error(errors, rn, "due_date", "Unparseable due date", row["due_date"], "error",
                          "Use ISO date YYYY-MM-DD.")
            elif not pd.isna(issue) and due < issue:
                add_error(errors, rn, "due_date", "Due date before issue date", due.date().isoformat(), "error",
                          "Correct due date.")
            elif not pd.isna(due) and due > today_ts + timedelta(days=365):
                add_error(errors, rn, "due_date", "Due date more than one year out", due.date().isoformat(), "warning",
                          "Verify contract terms.")
            if pd.isna(amount) or amount <= 0:
                add_error(errors, rn, "amount", "Missing, zero, or negative amount", row["amount"], "error",
                          "Amount must be positive.")
            elif amount > MAX_AMOUNT:
                add_error(errors, rn, "amount", "Amount out of range", amount, "error",
                          f"Maximum allowed is {MAX_AMOUNT:,.0f}.")
            elif amount > 1_000_000:
                add_error(errors, rn, "amount", "Large amount", amount, "warning",
                          "Require supporting evidence.")
            if currency not in ALLOWED_CURRENCIES:
                add_error(errors, rn, "currency", "Unsupported currency", currency, "error",
                          "Use an ISO currency from the reference catalog.")
            if "description" not in df.columns or not get_val(row, "description"):
                add_error(errors, rn, "description", "Missing description", "", "warning",
                          "Add goods/services description.")

        dup_mask = df.duplicated(subset=["invoice_number", "debtor_id_resolved"], keep=False)
        for _, row in df[dup_mask].iterrows():
            add_error(errors, int(row["row_number"]), "invoice_number",
                      "Duplicate invoice number within file", row["invoice_number"], "error",
                      "Remove duplicate or correct invoice number/debtor.")

        existing_df = pd.read_sql_query("SELECT invoice_number, debtor_id FROM invoices", conn)
        existing = set()
        if not existing_df.empty:
            existing = set(zip(existing_df["invoice_number"].astype(str), existing_df["debtor_id"].astype(str)))
        for _, row in df.iterrows():
            key = (str(row["invoice_number"]), str(row["debtor_id_resolved"]))
            if key in existing:
                add_error(errors, int(row["row_number"]), "invoice_number",
                          "Duplicate invoice already exists in database", row["invoice_number"], "error",
                          "Use re-import correction workflow or new invoice number.")

    elif entity_type == "payments":
        required = ["payment_date", "amount", "currency"]
        missing = [c for c in required if c not in df.columns]
        if "invoice_id" not in df.columns and "invoice_number" not in df.columns:
            missing.append("invoice_id or invoice_number")
        if missing:
            for m in missing:
                add_error(errors, 0, m, "Required column missing", "", "error", f"Add column: {m}")
            return df, pd.DataFrame(errors, columns=error_columns)

        df["row_number"] = df.index + 2

        inv_df = pd.read_sql_query(
            "SELECT id, invoice_number, debtor_id, issue_date, amount, currency FROM invoices",
            conn
        )
        inv_by_id = inv_df.set_index("id").to_dict("index") if not inv_df.empty else {}
        inv_by_num_debtor = {}
        inv_by_num = {}
        if not inv_df.empty:
            for _, r in inv_df.iterrows():
                key = (str(r["invoice_number"]), str(r["debtor_id"]))
                inv_by_num_debtor[key] = str(r["id"])
                inv_by_num.setdefault(str(r["invoice_number"]), []).append(str(r["id"]))

        debtor_rows = pd.read_sql_query("SELECT id, name FROM organizations", conn)
        name_to_id = {
            str(n).strip().lower(): str(i)
            for i, n in zip(debtor_rows["id"], debtor_rows["name"])
        } if not debtor_rows.empty else {}

        resolved_ids: List[str] = []
        resolved_inv_numbers: List[str] = []
        resolved_debtor_ids: List[str] = []

        for _, row in df.iterrows():
            rn = int(row["row_number"])
            inv_id = get_val(row, "invoice_id")
            inv_no = get_val(row, "invoice_number")
            d_id = get_val(row, "debtor_id")
            d_name = get_val(row, "debtor_name")
            if not d_id and d_name:
                d_id = name_to_id.get(d_name.lower(), "")

            if inv_id and inv_id in inv_by_id:
                pass
            elif inv_no:
                if d_id:
                    inv_id = inv_by_num_debtor.get((inv_no, d_id), "")
                    if not inv_id:
                        add_error(errors, rn, "invoice_number", "Unknown invoice for debtor", inv_no, "error",
                                  "Correct invoice_number/debtor_id.")
                else:
                    candidates = inv_by_num.get(inv_no, [])
                    if len(candidates) == 1:
                        inv_id = candidates[0]
                    elif len(candidates) > 1:
                        add_error(errors, rn, "invoice_number", "Ambiguous invoice number", inv_no, "error",
                                  "Provide debtor_id or invoice_id.")
                    else:
                        add_error(errors, rn, "invoice_number", "Unknown invoice number", inv_no, "error",
                                  "Import invoice first or correct reference.")
            else:
                add_error(errors, rn, "invoice_id", "Missing invoice reference", "", "error",
                          "Provide invoice_id or invoice_number.")

            resolved_ids.append(inv_id)
            info = inv_by_id.get(inv_id, {})
            resolved_inv_numbers.append(str(info.get("invoice_number", "")))
            resolved_debtor_ids.append(str(info.get("debtor_id", "")))

        df["invoice_id_resolved"] = resolved_ids
        df["invoice_number_resolved"] = resolved_inv_numbers
        df["debtor_id_resolved"] = resolved_debtor_ids
        df["payment_date_dt"] = pd.to_datetime(df["payment_date"], errors="coerce")
        df["amount_num"] = pd.to_numeric(df["amount"], errors="coerce")
        df["currency_clean"] = df["currency"].astype(str).str.upper().str.strip()
        df["reference_clean"] = df["reference"].astype(str).str.strip() if "reference" in df.columns else ""

        for _, row in df.iterrows():
            rn = int(row["row_number"])
            pay_dt = row["payment_date_dt"]
            amount = row["amount_num"]
            currency = row["currency_clean"]
            inv_id = row["invoice_id_resolved"]

            if pd.isna(pay_dt):
                add_error(errors, rn, "payment_date", "Unparseable payment date", row["payment_date"], "error",
                          "Use ISO date YYYY-MM-DD.")
            elif pay_dt.date() > today:
                add_error(errors, rn, "payment_date", "Future payment date", pay_dt.date().isoformat(), "error",
                          "Payment date cannot be after today.")

            if pd.isna(amount) or amount <= 0:
                add_error(errors, rn, "amount", "Missing, zero, or negative payment amount", row["amount"], "error",
                          "Payment amount must be positive.")

            if currency not in ALLOWED_CURRENCIES:
                add_error(errors, rn, "currency", "Unsupported currency", currency, "error",
                          "Use an ISO currency from the reference catalog.")

            if inv_id and inv_id in inv_by_id:
                info = inv_by_id[inv_id]
                inv_issue = pd.to_datetime(info.get("issue_date"), errors="coerce")
                inv_amount = safe_float(info.get("amount"))
                inv_currency = str(info.get("currency") or "").upper()
                if not pd.isna(pay_dt) and not pd.isna(inv_issue) and pay_dt < inv_issue:
                    add_error(errors, rn, "payment_date", "Payment before invoice issue date",
                              pay_dt.date().isoformat(), "error", "Correct payment date.")
                if currency and inv_currency and currency != inv_currency:
                    add_error(errors, rn, "currency", "Payment currency differs from invoice currency",
                              currency, "warning", "Verify FX or posting currency.")
                if np.isfinite(inv_amount) and inv_amount > 0 and np.isfinite(amount) and amount > inv_amount * 1.02:
                    add_error(errors, rn, "amount", "Single payment exceeds invoice amount", amount, "error",
                              "Correct amount or split payments.")

        new_sums = df[df["invoice_id_resolved"] != ""].groupby("invoice_id_resolved")["amount_num"].sum()
        payments_db = pd.read_sql_query("SELECT invoice_id, amount FROM payments", conn)
        existing_sums = payments_db.groupby("invoice_id")["amount"].sum() if not payments_db.empty else pd.Series(dtype=float)

        for inv_id, new_sum in new_sums.items():
            info = inv_by_id.get(inv_id, {})
            inv_amount = safe_float(info.get("amount"))
            existing = float(existing_sums.get(inv_id, 0.0))
            total = existing + float(new_sum)
            if np.isfinite(inv_amount) and inv_amount > 0 and total > inv_amount * 1.02:
                for _, row in df[df["invoice_id_resolved"] == inv_id].iterrows():
                    add_error(errors, int(row["row_number"]), "amount",
                              "Cumulative payments exceed invoice amount", total, "error",
                              "Correct payment amounts.")

        if "reference" in df.columns:
            ref_mask = (df["reference_clean"] != "") & df.duplicated(subset=["invoice_id_resolved", "reference_clean"], keep=False)
            for _, row in df[ref_mask].iterrows():
                add_error(errors, int(row["row_number"]), "reference",
                          "Duplicate payment reference within file", row["reference_clean"], "error",
                          "Use unique reference per invoice payment.")

            existing_refs_df = pd.read_sql_query("SELECT invoice_id, reference FROM payments", conn)
            existing_refs = set()
            if not existing_refs_df.empty:
                existing_refs = set(zip(existing_refs_df["invoice_id"].astype(str), existing_refs_df["reference"].fillna("").astype(str)))
            for _, row in df.iterrows():
                ref = str(row.get("reference_clean", ""))
                if ref and (str(row["invoice_id_resolved"]), ref) in existing_refs:
                    add_error(errors, int(row["row_number"]), "reference",
                              "Duplicate payment reference in database", ref, "error",
                              "Correct reference or use correction workflow.")

    elif entity_type == "debtors":
        required = ["debtor_id", "name", "country", "currency", "credit_limit"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            for m in missing:
                add_error(errors, 0, m, "Required column missing", "", "error", f"Add column: {m}")
            return df, pd.DataFrame(errors, columns=error_columns)

        df["row_number"] = df.index + 2
        df["currency_clean"] = df["currency"].astype(str).str.upper().str.strip()
        df["credit_limit_num"] = pd.to_numeric(df["credit_limit"], errors="coerce")

        existing_df = pd.read_sql_query("SELECT id FROM organizations", conn)
        existing_ids = set(existing_df["id"].astype(str)) if not existing_df.empty else set()

        dup_mask = df.duplicated(subset=["debtor_id"], keep=False)
        for _, row in df[dup_mask].iterrows():
            add_error(errors, int(row["row_number"]), "debtor_id",
                      "Duplicate debtor id within file", row["debtor_id"], "error",
                      "Use unique debtor_id.")

        for _, row in df.iterrows():
            rn = int(row["row_number"])
            did = get_val(row, "debtor_id")
            name = get_val(row, "name")
            country = get_val(row, "country")
            currency = row["currency_clean"]
            limit = row["credit_limit_num"]

            if not did:
                add_error(errors, rn, "debtor_id", "Missing debtor id", "", "error", "Provide debtor_id.")
            if did in existing_ids:
                add_error(errors, rn, "debtor_id", "Debtor already exists", did, "error",
                          "Use update workflow in post-MVP; MVP rejects duplicates.")
            if not name:
                add_error(errors, rn, "name", "Missing debtor name", "", "error", "Provide legal/trading name.")
            if not country:
                add_error(errors, rn, "country", "Missing country", "", "error", "Provide ISO country code.")
            if currency not in ALLOWED_CURRENCIES:
                add_error(errors, rn, "currency", "Unsupported currency", currency, "error",
                          "Use allowed ISO currency.")
            if pd.isna(limit) or limit < 0:
                add_error(errors, rn, "credit_limit", "Invalid credit limit", row["credit_limit"], "error",
                          "Credit limit must be zero or positive.")

    else:
        add_error(errors, 0, "entity_type", "Unsupported entity type", entity_type, "error",
                  "Use invoices, payments, or debtors.")

    return df, pd.DataFrame(errors, columns=error_columns)


def perform_import(conn: sqlite3.Connection, df: pd.DataFrame, entity_type: str,
                   upload_id: str, filename: str, actor: str, role: str) -> int:
    inserted = 0
    now = utcnow().isoformat()

    if entity_type == "invoices":
        for _, r in df.iterrows():
            inv_id = uid()
            conn.execute(
                """
                INSERT INTO invoices
                    (id, invoice_number, debtor_id, debtor_name, issue_date, due_date,
                     amount, currency, description, lifecycle_status, source,
                     source_file_id, row_number, created_at, updated_at, schema_version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    inv_id,
                    str(r["invoice_number"]).strip(),
                    str(r["debtor_id_resolved"]).strip(),
                    "",
                    to_date(r["issue_date_dt"]).isoformat(),
                    to_date(r["due_date_dt"]).isoformat(),
                    float(r["amount_num"]),
                    str(r["currency_clean"]).strip(),
                    get_val(r, "description"),
                    "validated",
                    filename,
                    upload_id,
                    int(r["row_number"]),
                    now,
                    now,
                    SCHEMA_VERSION,
                ),
            )
            inserted += 1

    elif entity_type == "payments":
        for _, r in df.iterrows():
            pay_id = uid()
            conn.execute(
                """
                INSERT INTO payments
                    (id, invoice_id, invoice_number, debtor_id, payment_date,
                     amount, currency, reference, source, source_file_id,
                     row_number, created_at, schema_version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    pay_id,
                    str(r["invoice_id_resolved"]).strip(),
                    str(r["invoice_number_resolved"]).strip(),
                    str(r["debtor_id_resolved"]).strip(),
                    to_date(r["payment_date_dt"]).isoformat(),
                    float(r["amount_num"]),
                    str(r["currency_clean"]).strip(),
                    str(r.get("reference_clean", "")),
                    filename,
                    upload_id,
                    int(r["row_number"]),
                    now,
                    SCHEMA_VERSION,
                ),
            )
            inserted += 1

    elif entity_type == "debtors":
        for _, r in df.iterrows():
            org_id = str(r["debtor_id"]).strip()
            conn.execute(
                """
                INSERT INTO organizations
                    (id, name, country, currency, sector, credit_limit,
                     created_at, updated_at, source, schema_version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    org_id,
                    str(r["name"]).strip(),
                    str(r["country"]).strip(),
                    str(r["currency_clean"]).strip(),
                    get_val(r, "sector"),
                    float(r["credit_limit_num"]),
                    now,
                    now,
                    filename,
                    SCHEMA_VERSION,
                ),
            )
            inserted += 1

    audit(conn, actor, role, "import", entity_type, upload_id,
          {"filename": filename, "inserted": inserted})
    conn.commit()
    return inserted


@st.cache_data(show_spinner=False)
def sample_file_bytes(entity: str, invalid: bool = False) -> bytes:
    if entity == "invoices" and not invalid:
        df = pd.DataFrame([
            {"invoice_number": "IMP-0001", "debtor_id": "D-1001", "issue_date": "2026-09-15",
             "due_date": "2026-10-15", "amount": 25000, "currency": "USD", "description": "Software subscription"},
            {"invoice_number": "IMP-0002", "debtor_id": "D-1002", "issue_date": "2026-09-20",
             "due_date": "2026-11-05", "amount": 42000, "currency": "EUR", "description": "Components supply"},
            {"invoice_number": "IMP-0003", "debtor_id": "D-1003", "issue_date": "2026-10-01",
             "due_date": "2026-10-31", "amount": 18000, "currency": "SGD", "description": "Logistics services"},
        ])
    elif entity == "invoices" and invalid:
        df = pd.DataFrame([
            {"invoice_number": "BAD-0001", "debtor_id": "D-1001", "issue_date": "2026-10-01",
             "due_date": "2026-10-31", "amount": -500, "currency": "USD", "description": "Negative amount"},
            {"invoice_number": "BAD-0002", "debtor_id": "D-9999", "issue_date": "2026-10-02",
             "due_date": "2026-11-02", "amount": 10000, "currency": "USD", "description": "Unknown debtor"},
            {"invoice_number": "BAD-0003", "debtor_id": "D-1002", "issue_date": "2030-01-01",
             "due_date": "2030-02-01", "amount": 9000, "currency": "EUR", "description": "Future issue date"},
            {"invoice_number": "BAD-0004", "debtor_id": "D-1003", "issue_date": "2026-10-05",
             "due_date": "2026-10-01", "amount": 7000, "currency": "SGD", "description": "Due before issue"},
            {"invoice_number": "BAD-0005", "debtor_id": "D-1003", "issue_date": "2026-10-05",
             "due_date": "2026-11-05", "amount": 7000, "currency": "XXX", "description": "Unsupported currency"},
            {"invoice_number": "BAD-0006", "debtor_id": "D-1001", "issue_date": "2026-10-06",
             "due_date": "2026-11-06", "amount": 12000, "currency": "USD", "description": "Duplicate row A"},
            {"invoice_number": "BAD-0006", "debtor_id": "D-1001", "issue_date": "2026-10-06",
             "due_date": "2026-11-06", "amount": 12000, "currency": "USD", "description": "Duplicate row B"},
        ])
    elif entity == "payments":
        df = pd.DataFrame([
            {"invoice_number": "IMP-0001", "debtor_id": "D-1001", "payment_date": "2026-10-17",
             "amount": 25000, "currency": "USD", "reference": "REM-0001"},
            {"invoice_number": "IMP-0002", "debtor_id": "D-1002", "payment_date": "2026-11-08",
             "amount": 21000, "currency": "EUR", "reference": "REM-0002A"},
            {"invoice_number": "IMP-0002", "debtor_id": "D-1002", "payment_date": "2026-11-20",
             "amount": 21000, "currency": "EUR", "reference": "REM-0002B"},
        ])
    else:
        df = pd.DataFrame([
            {"debtor_id": "D-2001", "name": "Demo New Debtor One", "country": "US",
             "currency": "USD", "sector": "Retail", "credit_limit": 250000},
            {"debtor_id": "D-2002", "name": "Demo New Debtor Two", "country": "DE",
             "currency": "EUR", "sector": "Manufacturing", "credit_limit": 300000},
        ])
    return df.to_csv(index=False).encode("utf-8")


def init_simulation() -> None:
    st.session_state.sim_start_date = utcnow().date() - timedelta(days=120)
    st.session_state.sim_day_index = 0
    st.session_state.sim_history = []
    st.session_state.sim_running = True
    st.session_state.sim_step_request = 0
    st.session_state.sim_speed = 1
    for _ in range(20):
        advance_simulation()


def ensure_simulation() -> None:
    if "sim_history" not in st.session_state:
        init_simulation()


def advance_simulation() -> None:
    ensure_simulation()
    day_index = int(st.session_state.get("sim_day_index", 0))
    start = st.session_state.get("sim_start_date", utcnow().date() - timedelta(days=120))
    as_of = pd.Timestamp(start + timedelta(days=day_index))
    metrics, _ = calculate_portfolio_metrics(as_of)
    buckets = metrics["aging_buckets"]

    row = {
        "date": as_of.date().isoformat(),
        "outstanding": float(metrics["total_outstanding"]),
        "overdue": float(metrics["overdue_amount"]),
        "risk_score": float(metrics["portfolio_risk"]),
        "validity": float(metrics["avg_validity"]),
        "top1_share": float(metrics["top1_share"]),
        "hhi": float(metrics["hhi"]),
        "avg_delay": float(metrics["avg_delay_days"]),
        "current": float(buckets.get("Current", 0.0)),
        "bucket_1_30": float(buckets.get("1-30", 0.0)),
        "bucket_31_60": float(buckets.get("31-60", 0.0)),
        "bucket_61_90": float(buckets.get("61-90", 0.0)),
        "bucket_90_plus": float(buckets.get("90+", 0.0)),
    }

    st.session_state.sim_history.append(row)
    if len(st.session_state.sim_history) > 180:
        st.session_state.sim_history.pop(0)

    day_index += 1
    if day_index > 120:
        day_index = 0
    st.session_state.sim_day_index = day_index


def draw_progress_line(frame, x, y, w, h, full_values, upto_values, color, label, vmin, vmax):
    cv2.rectangle(frame, (x, y), (x + w, y + h), (32, 36, 44), -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (70, 75, 85), 1)
    cv2.putText(frame, label, (x + 10, y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
    if vmax <= vmin:
        vmax = vmin + 1.0
    n_full = max(1, len(full_values) - 1)
    pts = []
    for j, val in enumerate(upto_values):
        px = x + 55 + int((w - 80) * (j / n_full))
        py = y + h - 35 - int((h - 70) * ((float(val) - vmin) / (vmax - vmin)))
        pts.append((px, py))
    cv2.line(frame, (x + 50, y + h - 35), (x + w - 20, y + h - 35), (90, 95, 105), 1)
    cv2.line(frame, (x + 50, y + 35), (x + 50, y + h - 35), (90, 95, 105), 1)
    if len(pts) > 1:
        cv2.polylines(frame, [np.array(pts, dtype=np.int32)], False, color, 2)
    elif len(pts) == 1:
        cv2.circle(frame, pts[0], 3, color, -1)


def draw_bars(frame, x, y, w, h, values, labels):
    cv2.rectangle(frame, (x, y), (x + w, y + h), (32, 36, 44), -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (70, 75, 85), 1)
    cv2.putText(frame, "Aging buckets", (x + 10, y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
    maxv = max(float(max(values)), 1.0)
    n = len(values)
    slot = w / (n + 1)
    colors = [(0, 200, 120), (0, 180, 255), (0, 140, 255), (0, 80, 255), (0, 0, 255)]
    for i, val in enumerate(values):
        bx = int(x + slot * (i + 0.65))
        bh = int((h - 70) * (float(val) / maxv))
        by1 = y + h - 40
        by2 = by1 - bh
        cv2.rectangle(frame, (bx, by2), (bx + int(slot * 0.55), by1), colors[i % len(colors)], -1)
        cv2.putText(frame, labels[i], (bx - 5, by1 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (190, 190, 190), 1)


def generate_demo_video_bytes() -> Optional[bytes]:
    if cv2 is None:
        return None

    width, height, fps, seconds = 960, 540, 15, 8
    frame_count = max(2, fps * seconds)
    tmp_path = os.path.join(tempfile.gettempdir(), f"ifri_demo_{uuid.uuid4().hex}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_path, fourcc, fps, (width, height))

    if not writer.isOpened():
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        return None

    try:
        rng = np.random.default_rng(2026)
        t = np.linspace(0.0, 1.0, frame_count)

        overdue = 250000 + 170000 * t + rng.normal(0, 9000, frame_count)
        overdue = np.clip(overdue, 50000, 1000000)
        current_extra = 180000 + 40000 * np.sin(t * 6.0) + rng.normal(0, 5000, frame_count)
        current_extra = np.clip(current_extra, 50000, 500000)
        outstanding_total = current_extra + overdue

        risk = 38 + 24 * t + rng.normal(0, 2.5, frame_count)
        risk = np.clip(risk, 0, 100)
        validity = 92 - 12 * t + rng.normal(0, 1.8, frame_count)
        validity = np.clip(validity, 0, 100)
        conc = 0.29 + 0.16 * t + rng.normal(0, 0.012, frame_count)
        conc = np.clip(conc, 0, 1)

        props = np.tile(np.array([0.45, 0.25, 0.15, 0.10, 0.05]), (frame_count, 1))
        props += (t[:, None] - 0.5) * np.array([-0.12, -0.06, 0.0, 0.06, 0.12])
        props = np.clip(props, 0.04, 0.65)
        props /= props.sum(axis=1, keepdims=True)
        aging = overdue[:, None] * props

        vmin = float(overdue.min())
        vmax = float(overdue.max())

        for i in range(frame_count):
            frame = np.full((height, width, 3), (24, 26, 32), dtype=np.uint8)

            cv2.rectangle(frame, (0, 0), (width, 70), (40, 45, 55), -1)
            cv2.putText(frame, "Invoice Financing & Receivables Risk Intelligence",
                        (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (235, 235, 235), 2)
            cv2.putText(frame, "Synthetic live demo | Human-reviewed decision support | Not autonomous lending",
                        (20, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (170, 190, 210), 1)

            cards = [
                ("Outstanding", f"${outstanding_total[i]:,.0f}", (255, 180, 80)),
                ("Overdue", f"${overdue[i]:,.0f}", (255, 160, 0)),
                ("Portfolio risk", f"{risk[i]:.1f}", (0, 80, 255) if risk[i] > 60 else (0, 200, 120)),
                ("Validity", f"{validity[i]:.1f}", (180, 220, 255)),
            ]
            x = 30
            for label, value, color in cards:
                cv2.rectangle(frame, (x, 100), (x + 205, 170), (35, 40, 48), -1)
                cv2.rectangle(frame, (x, 100), (x + 205, 170), color, 2)
                cv2.putText(frame, label, (x + 10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (190, 190, 190), 1)
                cv2.putText(frame, value, (x + 10, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.68, color, 2)
                x += 225

            draw_progress_line(frame, 35, 200, 520, 245, overdue, overdue[:i + 1],
                               (0, 180, 255), "Overdue receivables trend", vmin, vmax)

            draw_bars(frame, 590, 200, 340, 245, [float(v) for v in aging[i]],
                      ["Cur", "1-30", "31-60", "61-90", "90+"])

            cv2.putText(frame, f"Top-1 debtor share: {conc[i] * 100:.1f}%",
                        (35, 475), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (210, 210, 210), 1)
            cv2.putText(frame, "All financial outputs require authorized human review.",
                        (35, 500), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (150, 200, 255), 1)

            cv2.rectangle(frame, (30, 515), (width - 30, 532), (55, 60, 70), -1)
            prog = int((width - 60) * (i / max(1, frame_count - 1)))
            cv2.rectangle(frame, (30, 515), (30 + prog, 532), (0, 160, 255), -1)

            writer.write(frame)

        writer.release()
        with open(tmp_path, "rb") as f:
            data = f.read()
        return data if len(data) > 1000 else None

    except Exception:
        try:
            writer.release()
        except Exception:
            pass
        return None

    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


def render_live_dashboard_body() -> None:
    ensure_simulation()

    if st.session_state.get("sim_running", False):
        speed = max(1, int(st.session_state.get("sim_speed", 1)))
        for _ in range(speed):
            advance_simulation()
    elif int(st.session_state.get("sim_step_request", 0)) > 0:
        steps = int(st.session_state["sim_step_request"])
        for _ in range(steps):
            advance_simulation()
        st.session_state.sim_step_request = 0

    hist = st.session_state.get("sim_history", [])
    if not hist:
        st.info("Initializing simulation...")
        return

    df = pd.DataFrame(hist)
    latest = df.iloc[-1]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Sim date", latest["date"])
    c2.metric("Outstanding", f"${float(latest['outstanding']):,.0f}")
    c3.metric("Overdue", f"${float(latest['overdue']):,.0f}")
    c4.metric("Risk score", f"{float(latest['risk_score']):.1f}")
    c5.metric("Avg validity", f"{float(latest['validity']):.1f}")

    fig = px.line(
        df,
        x="date",
        y=["outstanding", "overdue"],
        title="Live receivables exposure (synthetic simulation)",
        labels={"value": "Amount", "variable": "Series", "date": "Simulation date"},
    )
    fig.update_layout(height=300, legend_title="")
    st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)
    with left:
        aging_df = pd.DataFrame({
            "Bucket": BUCKET_ORDER,
            "Amount": [
                float(latest["current"]),
                float(latest["bucket_1_30"]),
                float(latest["bucket_31_60"]),
                float(latest["bucket_61_90"]),
                float(latest["bucket_90_plus"]),
            ],
        })
        fig2 = px.bar(aging_df, x="Bucket", y="Amount", title="Aging buckets at latest step")
        fig2.update_layout(height=280)
        st.plotly_chart(fig2, use_container_width=True)

    with right:
        fig3 = go.Figure(go.Indicator(
            mode="gauge+number",
            value=float(latest["risk_score"]),
            title={"text": "Portfolio risk score"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "crimson"},
                "steps": [
                    {"range": [0, 45], "color": "#d6f5d6"},
                    {"range": [45, 70], "color": "#fff3cd"},
                    {"range": [70, 100], "color": "#f8d7da"},
                ],
                "threshold": {"line": {"color": "black", "width": 3}, "thickness": 0.75, "value": 70},
            },
        ))
        fig3.update_layout(height=280)
        st.plotly_chart(fig3, use_container_width=True)

    fig4 = px.line(df, x="date", y=["top1_share", "hhi"], title="Debtor concentration trend")
    fig4.update_layout(height=260)
    st.plotly_chart(fig4, use_container_width=True)


def call_live_fragment() -> None:
    def body():
        render_live_dashboard_body()

    if hasattr(st, "fragment"):
        try:
            frag = st.fragment(body, run_every=2.0)
            frag()
        except Exception:
            body()
    else:
        body()


def render_live_demo() -> None:
    st.title("Live Demo — Invoice Financing & Receivables Risk Intelligence")
    st.caption(
        "This demo uses clearly labeled synthetic data only. It is decision support for authorized human reviewers, "
        "not regulated financial advice and not an autonomous lending system."
    )

    ensure_simulation()

    c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 2])
    if c1.button("Play", use_container_width=True):
        st.session_state.sim_running = True
    if c2.button("Pause", use_container_width=True):
        st.session_state.sim_running = False
    if c3.button("Step", use_container_width=True):
        st.session_state.sim_step_request = int(st.session_state.get("sim_step_request", 0)) + 1
    if c4.button("Reset", use_container_width=True):
        init_simulation()
    st.session_state.sim_speed = c5.slider("Steps per update", min_value=1, max_value=5, value=int(st.session_state.get("sim_speed", 1)))

    call_live_fragment()

    st.divider()
    st.subheader("Generated working demo video")

    if "demo_video_bytes" not in st.session_state:
        with st.spinner("Generating a short MP4 demo video..."):
            st.session_state.demo_video_bytes = generate_demo_video_bytes()

    video_bytes = st.session_state.get("demo_video_bytes")
    if video_bytes:
        st.video(video_bytes, format="video/mp4")
        st.download_button(
            "Download demo video (MP4)",
            data=video_bytes,
            file_name="invoice_financing_demo.mp4",
            mime="video/mp4",
        )
    else:
        st.warning(
            "MP4 generation is unavailable in this environment. The live dashboard above still runs. "
            "Install the OpenCV dependency from requirements.txt to enable in-app video generation."
        )


def render_import() -> None:
    st.header("Data Import")
    st.caption("CSV and XLSX first. Original files are preserved outside relational rows. Validated records are never silently overwritten.")

    entity = st.selectbox("Entity type", ["invoices", "payments", "debtors"])
    col1, col2, col3 = st.columns(3)
    with col1:
        st.download_button("Valid invoices sample", sample_file_bytes("invoices", False), "valid_invoices_sample.csv", "text/csv")
    with col2:
        st.download_button("Invalid invoices sample", sample_file_bytes("invoices", True), "invalid_invoices_sample.csv", "text/csv")
    with col3:
        st.download_button("Payments sample", sample_file_bytes("payments", False), "payments_sample.csv", "text/csv")

    uploaded = st.file_uploader("Upload CSV or XLSX", type=["csv", "xlsx", "xls"])
    if not uploaded:
        st.info("Upload a file to preview and validate.")
        return

    data = uploaded.getvalue()
    filename = uploaded.name
    try:
        if filename.lower().endswith(".csv"):
            raw_df = pd.read_csv(io.BytesIO(data))
        else:
            xls = pd.ExcelFile(io.BytesIO(data))
            sheet = st.selectbox("Sheet", xls.sheet_names)
            raw_df = xls.parse(sheet_name=sheet)
    except Exception as e:
        st.error(f"Could not read file: {e}")
        return

    with get_conn() as conn:
        norm_df, issues = validate_import(raw_df, entity, conn)

    st.subheader("Preview")
    st.dataframe(norm_df.head(25), use_container_width=True)

    blocking = issues[issues["severity"] == "error"] if not issues.empty else issues
    warnings = issues[issues["severity"] == "warning"] if not issues.empty else issues

    if not blocking.empty:
        st.error(f"{len(blocking)} blocking validation issues. Malformed rows are rejected.")
        st.dataframe(issues, use_container_width=True)
        st.download_button(
            "Download error report",
            data=issues.to_csv(index=False).encode("utf-8"),
            file_name=f"{entity}_validation_errors.csv",
            mime="text/csv",
        )
        if st.button("Save rejected upload for audit"):
            role = st.session_state.get("user_role", "Read-only Stakeholder")
            user = st.session_state.get("user_name", "Demo User")
            upload_id, _, _ = save_original_file(data, filename, entity, user, role, "rejected")
            err_path = save_error_report(issues, upload_id)
            mark_upload(upload_id, "rejected", len(norm_df), 0, err_path)
            st.success("Rejected upload and error report saved for lineage and audit.")
    else:
        st.success("All rows passed blocking validation.")
        if not warnings.empty:
            st.warning(f"{len(warnings)} non-blocking warnings.")
            st.dataframe(warnings, use_container_width=True)

        role = st.session_state.get("user_role", "Read-only Stakeholder")
        user = st.session_state.get("user_name", "Demo User")
        if not has_perm(role, "import"):
            st.error("Current role does not have import permission.")
        else:
            if st.button("Import validated rows"):
                upload_id, _, _ = save_original_file(data, filename, entity, user, role, "pending")
                with get_conn() as conn:
                    inserted = perform_import(conn, norm_df, entity, upload_id, filename, user, role)
                mark_upload(upload_id, "imported", len(norm_df), inserted, None)
                st.success(f"Imported {inserted} rows. Source file preserved with checksum.")
                st.rerun()


def render_records() -> None:
    st.header("Records")

    with get_conn() as conn:
        invoices = pd.read_sql_query("SELECT * FROM invoices", conn)

    if invoices.empty:
        st.info("No records.")
        return

    statuses = sorted([s for s in invoices["lifecycle_status"].dropna().unique() if s])
    debtors = sorted([d for d in invoices["debtor_id"].dropna().unique() if d])

    f1, f2, f3 = st.columns(3)
    status_filter = f1.multiselect("Lifecycle status", statuses, default=statuses)
    debtor_filter = f2.multiselect("Debtor", debtors, default=debtors)
    min_risk = f3.slider("Minimum risk score", 0.0, 100.0, 0.0)

    filtered = invoices.copy()
    if status_filter:
        filtered = filtered[filtered["lifecycle_status"].isin(status_filter)]
    if debtor_filter:
        filtered = filtered[filtered["debtor_id"].isin(debtor_filter)]
    filtered["risk_score_num"] = pd.to_numeric(filtered["risk_score"], errors="coerce").fillna(0)
    filtered = filtered[filtered["risk_score_num"] >= min_risk]

    show_cols = [
        "invoice_number", "debtor_id", "debtor_name", "issue_date", "due_date",
        "amount", "currency", "lifecycle_status", "validity_score", "risk_score",
        "recommended_decision", "final_decision"
    ]
    st.dataframe(filtered[[c for c in show_cols if c in filtered.columns]], use_container_width=True)

    if filtered.empty:
        return

    selected_id = st.selectbox("Select invoice", filtered["id"].tolist(), format_func=lambda x: x)
    assessment = get_invoice_assessment(selected_id, utcnow().date())
    if assessment:
        st.subheader("Traceable explanation")
        st.json(json_ready(assessment))

    with get_conn() as conn:
        payments = pd.read_sql_query(
            "SELECT payment_date, amount, currency, reference, source FROM payments WHERE invoice_id = ?",
            conn,
            params=(selected_id,),
        )
    if not payments.empty:
        st.subheader("Payments")
        st.dataframe(payments, use_container_width=True)


def get_invoice_assessment(inv_id: str, as_of: date) -> Optional[Dict[str, Any]]:
    with get_conn() as conn:
        inv = conn.execute("SELECT * FROM invoices WHERE id = ?", (inv_id,)).fetchone()
        if not inv:
            return None
        debtor = conn.execute("SELECT * FROM organizations WHERE id = ?", (inv["debtor_id"],)).fetchone()
        payments = conn.execute("SELECT * FROM payments WHERE invoice_id = ?", (inv_id,)).fetchall()
        dup = conn.execute(
            "SELECT COUNT(*) AS c FROM invoices WHERE invoice_number = ? AND debtor_id = ?",
            (inv["invoice_number"], inv["debtor_id"]),
        ).fetchone()["c"]

    return compute_invoice_assessment(
        dict(inv),
        dict(debtor) if debtor else None,
        [dict(p) for p in payments],
        pd.Timestamp(as_of),
        int(dup or 1),
    )


def evaluate_synthetic_forecast(history_df: pd.DataFrame, horizon: int = 7) -> Optional[pd.DataFrame]:
    if history_df is None or len(history_df) < horizon * 3:
        return None
    series = history_df["overdue"].astype(float)
    train = series[:-horizon]
    test = series[-horizon:]
    if len(train) < horizon:
        return None
    ma_value = float(train.rolling(horizon).mean().iloc[-1])
    persist_value = float(train.iloc[-1])
    ma_pred = np.full(horizon, ma_value)
    persist_pred = np.full(horizon, persist_value)
    mae_ma = float(np.mean(np.abs(ma_pred - test.values)))
    mae_persist = float(np.mean(np.abs(persist_pred - test.values)))
    return pd.DataFrame({
        "Method": ["7-day moving average", "Persistence"],
        "MAE": [mae_ma, mae_persist],
        "Scope": ["Synthetic simulation only", "Synthetic simulation only"],
        "Limitation": ["Not a real-world forecast claim", "Not a real-world forecast claim"],
    })


def render_analysis() -> None:
    st.header("Risk Analysis")

    as_of = st.date_input("As of date", utcnow().date())
    metrics, scored = calculate_portfolio_metrics(as_of)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Outstanding", f"${metrics['total_outstanding']:,.0f}")
    c2.metric("Overdue", f"${metrics['overdue_amount']:,.0f}", f"{metrics['overdue_pct']:.1f}%")
    c3.metric("Avg validity", f"{metrics['avg_validity']:.1f}")
    c4.metric("Portfolio risk", f"{metrics['portfolio_risk']:.1f}")
    c5.metric("Top-1 share", f"{metrics['top1_share'] * 100:.1f}%")

    st.markdown("#### Deterministic formulas")
    st.code(
        f"""
Rules version: {metrics['rules_version']}
Validity score = max(0, 100 - sum(validity deductions))
Invoice risk = 0.6 * (100 - validity score) + 0.4 * aging risk
Aging risk = min(100, days_past_due / 120 * 100)
Portfolio risk = 0.35*validity_risk + 0.25*aging_risk + 0.20*concentration_risk + 0.20*delay_risk
Concentration risk = min(100, (top1_share*0.7 + top3_share*0.3)*100)
Delay risk = min(100, max(0, avg_delay/60*70) + pct_late*30)
""",
        language="text",
    )

    left, right = st.columns(2)
    with left:
        aging_df = pd.DataFrame({
            "Bucket": list(metrics["aging_buckets"].keys()),
            "Amount": list(metrics["aging_buckets"].values()),
        })
        fig = px.bar(aging_df, x="Bucket", y="Amount", title="Receivables aging")
        fig.update_layout(height=320)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        top_df = pd.DataFrame(metrics["top_debtors"])
        if not top_df.empty:
            fig2 = px.bar(top_df, x="name", y="exposure", title="Debtor exposure concentration")
            fig2.update_layout(height=320)
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No debtor exposure at this as-of date.")

    if "sim_history" in st.session_state and st.session_state.sim_history:
        hist_df = pd.DataFrame(st.session_state.sim_history)
        forecast_eval = evaluate_synthetic_forecast(hist_df)
        if forecast_eval is not None:
            st.subheader("Synthetic baseline forecast evaluation")
            st.caption("Evaluation uses synthetic simulation data only. It validates software plumbing, not real-world predictive performance.")
            st.dataframe(forecast_eval, use_container_width=True)

    st.subheader("Highest-risk invoices")
    risk_cols = ["invoice_number", "debtor_name", "due_date", "amount", "outstanding",
                 "validity_score", "risk_score", "aging_bucket", "recommended_decision",
                 "lifecycle_status", "final_decision"]
    risk_view = scored.sort_values("risk_score", ascending=False)[[c for c in risk_cols if c in scored.columns]].head(25)
    st.dataframe(risk_view, use_container_width=True)

    selected_id = st.selectbox("Explain invoice", scored["id"].tolist(), key="analysis_explain_id")
    row = scored[scored["id"] == selected_id].iloc[0]
    st.subheader("Explanation panel")
    explanation = {
        "input_values": {
            "invoice_number": row["invoice_number"],
            "debtor_id": row["debtor_id"],
            "debtor_name": row["debtor_name"],
            "issue_date": str(row["issue_date"]),
            "due_date": str(row["due_date"]),
            "amount": float(row["amount"]) if pd.notna(row["amount"]) else None,
            "paid_amount": float(row["paid_amount"]),
            "outstanding": float(row["outstanding"]),
            "currency": row["currency"],
        },
        "rule_version": row["rules_version"],
        "validity_score": float(row["validity_score"]),
        "risk_score": float(row["risk_score"]),
        "flags": row["flags"],
        "hard_flags": row["hard_flags"],
        "key_contributing_factors": row["contributors"],
        "data_quality_warnings": row["warnings"],
        "recommended_decision": row["recommended_decision"],
        "reviewer_status": {
            "lifecycle_status": row["lifecycle_status"],
            "final_decision": row["final_decision"],
            "reviewed_by": row["reviewed_by"],
            "reviewed_at": row["reviewed_at"],
        },
        "uncertainty_note": "This MVP uses deterministic rules. Uncertainty is represented as data-quality completeness and explicit human-review status, not statistical confidence.",
        "human_approval_boundary": "No autonomous financing decision is made. Authorized reviewer approval is required.",
    }
    st.json(json_ready(explanation))

    role = st.session_state.get("user_role", "Read-only Stakeholder")
    if has_perm(role, "calculate"):
        if st.button("Refresh stored scores for this as-of date"):
            refresh_stored_scores(as_of)
            st.success("Stored invoice validity and risk scores refreshed.")


def compute_financing_cap(row: pd.Series, scored: pd.DataFrame, as_of_ts: pd.Timestamp) -> Tuple[float, float, float]:
    outstanding = float(row["outstanding"])
    risk = float(row["risk_score"])
    if risk <= 30:
        rate = 0.90
    elif risk <= 50:
        rate = 0.80
    elif risk <= 70:
        rate = 0.65
    else:
        rate = 0.0

    debtor_exposure = float(
        scored[(scored["debtor_id"] == row["debtor_id"]) & (scored["issue_date"] <= as_of_ts)]["outstanding"].sum()
    )
    credit_limit = safe_float(row["credit_limit"])
    if np.isfinite(credit_limit) and credit_limit > 0:
        remaining = max(0.0, credit_limit - debtor_exposure)
    else:
        remaining = outstanding

    cap = min(outstanding * rate, remaining)
    return rate, cap, remaining


def record_decision(inv_id: str, decision: str, requested: float, approved: float,
                    risk_score: float, rationale: str, override_reason: str,
                    actor: str, role: str, new_status: str) -> None:
    with get_conn() as conn:
        conn.execute(
            """
            UPDATE invoices
            SET lifecycle_status = ?, final_decision = ?, reviewed_by = ?,
                reviewed_at = ?, override_reason = ?, updated_at = ?
            WHERE id = ?
            """,
            (
                new_status,
                decision,
                actor,
                utcnow().isoformat(),
                override_reason,
                utcnow().isoformat(),
                inv_id,
            ),
        )
        conn.execute(
            """
            INSERT INTO decisions
                (id, entity_type, entity_id, decision, requested_amount, approved_amount,
                 risk_score, rationale, override_reason, actor, role, created_at, schema_version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                uid(),
                "invoice",
                inv_id,
                decision,
                float(requested),
                float(approved),
                float(risk_score),
                rationale,
                override_reason,
                actor,
                role,
                utcnow().isoformat(),
                SCHEMA_VERSION,
            ),
        )
        audit(conn, actor, role, decision.lower(), "invoice", inv_id,
              {"requested": requested, "approved": approved, "override_reason": override_reason})
        conn.commit()


def render_review() -> None:
    st.header("Review & Approvals")
    st.caption("Human-reviewed financing decisions. Overrides require reason and are audited.")

    as_of = st.date_input("Review as-of date", utcnow().date(), key="review_as_of")
    as_of_ts = pd.Timestamp(as_of)
    metrics, scored = calculate_portfolio_metrics(as_of)

    queue_statuses = ["validated", "exception", "under_review", "corrected"]
    queue = scored[scored["lifecycle_status"].isin(queue_statuses)].copy()
    queue = queue.sort_values("risk_score", ascending=False)

    if queue.empty:
        st.info("No items in review queue.")
        return

    selected_id = st.selectbox(
        "Queue item",
        queue["id"].tolist(),
        format_func=lambda x: f"{queue[queue['id'] == x].iloc[0]['invoice_number']} | risk {queue[queue['id'] == x].iloc[0]['risk_score']:.1f}",
    )
    row = queue[queue["id"] == selected_id].iloc[0]

    rate, cap, remaining = compute_financing_cap(row, scored, as_of_ts)

    st.subheader("Item detail")
    detail = {
        "invoice_number": row["invoice_number"],
        "debtor": row["debtor_name"],
        "amount": float(row["amount"]),
        "outstanding": float(row["outstanding"]),
        "aging_bucket": row["aging_bucket"],
        "validity_score": float(row["validity_score"]),
        "risk_score": float(row["risk_score"]),
        "recommended_decision": row["recommended_decision"],
        "flags": row["flags"],
        "warnings": row["warnings"],
        "advance_rate": rate,
        "financing_cap": cap,
        "debtor_remaining_limit": remaining,
        "lifecycle_status": row["lifecycle_status"],
    }
    st.json(json_ready(detail))

    user = st.session_state.get("user_name", "Demo User")
    role = st.session_state.get("user_role", "Read-only Stakeholder")

    requested = st.number_input("Requested financing amount", min_value=0.0, value=float(min(cap, row["outstanding"])), step=100.0)
    override_reason = st.text_input("Override reason (required if overriding recommendation)")

    col1, col2, col3, col4 = st.columns(4)

    if col1.button("Submit for review"):
        if not has_perm(role, "prepare_review"):
            st.error("Permission denied.")
        else:
            with get_conn() as conn:
                conn.execute(
                    "UPDATE invoices SET lifecycle_status = ?, updated_at = ? WHERE id = ?",
                    ("under_review", utcnow().isoformat(), selected_id),
                )
                audit(conn, user, role, "submit_review", "invoice", selected_id, {})
                conn.commit()
            st.success("Submitted for review.")
            st.rerun()

    if col2.button("Approve"):
        if not has_perm(role, "approve"):
            st.error("Permission denied.")
        elif row["lifecycle_status"] not in ["under_review", "corrected"] and role != "Administrator":
            st.error("Item must be submitted for review before approval.")
        elif row["recommended_decision"] == "Reject" and not override_reason.strip():
            st.error("Override reason is required to approve a rejected recommendation.")
        elif not has_perm(role, "override") and row["recommended_decision"] == "Reject":
            st.error("Permission denied for override.")
        else:
            rationale = json.dumps({
                "risk_score": float(row["risk_score"]),
                "validity_score": float(row["validity_score"]),
                "recommended": row["recommended_decision"],
                "cap": float(cap),
                "rules_version": RULES_VERSION,
            }, default=str)
            record_decision(selected_id, "Approved", requested, min(requested, cap),
                            float(row["risk_score"]), rationale, override_reason, user, role, "approved")
            st.success("Approved with audit trail.")
            st.rerun()

    if col3.button("Reject"):
        if not has_perm(role, "approve"):
            st.error("Permission denied.")
        else:
            rationale = json.dumps({
                "risk_score": float(row["risk_score"]),
                "recommended": row["recommended_decision"],
                "flags": row["flags"],
                "rules_version": RULES_VERSION,
            }, default=str)
            record_decision(selected_id, "Rejected", requested, 0.0,
                            float(row["risk_score"]), rationale, override_reason, user, role, "rejected")
            st.success("Rejected with audit trail.")
            st.rerun()

    if col4.button("Mark corrected"):
        if not has_perm(role, "prepare_review"):
            st.error("Permission denied.")
        elif not override_reason.strip():
            st.error("Correction reason is required.")
        else:
            with get_conn() as conn:
                conn.execute(
                    "UPDATE invoices SET lifecycle_status = ?, override_reason = ?, updated_at = ? WHERE id = ?",
                    ("corrected", override_reason, utcnow().isoformat(), selected_id),
                )
                audit(conn, user, role, "mark_corrected", "invoice", selected_id, {"reason": override_reason})
                conn.commit()
            st.success("Marked corrected.")
            st.rerun()

    st.divider()
    st.subheader("Recent decisions")
    with get_conn() as conn:
        decisions = pd.read_sql_query("SELECT * FROM decisions ORDER BY created_at DESC LIMIT 25", conn)
    st.dataframe(decisions, use_container_width=True)


def render_reports() -> None:
    st.header("Reports and Exports")

    as_of = st.date_input("Report as-of date", utcnow().date(), key="report_as_of")
    metrics, scored = calculate_portfolio_metrics(as_of)

    top_risk = scored.sort_values("risk_score", ascending=False).head(10)
    export_cols = [
        "invoice_number", "debtor_id", "debtor_name", "issue_date", "due_date",
        "amount", "paid_amount", "outstanding", "aging_bucket", "validity_score",
        "risk_score", "recommended_decision", "lifecycle_status", "final_decision"
    ]
    scored_export = scored[[c for c in export_cols if c in scored.columns]].copy()
    scored_export["issue_date"] = scored_export["issue_date"].astype(str)
    scored_export["due_date"] = scored_export["due_date"].astype(str)

    md = f"""
# Invoice Financing and Receivables Risk Report

**As of:** {metrics['as_of']}  
**Rules version:** {metrics['rules_version']}  
**Schema version:** {metrics['schema_version']}  
**Data classification:** Synthetic demo data. No real customer data.  
**Decision boundary:** Decision support only. Authorized human review required. No autonomous lending.

## Portfolio summary

- Invoices in scope: {metrics['total_invoices']}
- Total invoiced amount: {metrics['total_amount']:,.2f}
- Outstanding receivables: {metrics['total_outstanding']:,.2f}
- Overdue receivables: {metrics['overdue_amount']:,.2f} ({metrics['overdue_pct']:.1f}%)
- Average invoice validity score: {metrics['avg_validity']:.1f}
- Portfolio risk score: {metrics['portfolio_risk']:.1f}
- Weighted days past due: {metrics['weighted_days_past_due']:.1f}
- Average payment delay: {metrics['avg_delay_days']:.1f} days
- Percent paid late: {metrics['pct_late'] * 100:.1f}%
- Top-1 debtor share: {metrics['top1_share'] * 100:.1f}%
- Top-3 debtor share: {metrics['top3_share'] * 100:.1f}%
- HHI: {metrics['hhi']:.4f} ({metrics['hhi_10k']:.1f} on 0–10,000 scale)

## Aging buckets

| Bucket | Amount |
|---|---:|
""" + "\n".join([f"| {k} | {v:,.2f} |" for k, v in metrics["aging_buckets"].items()]) + f"""

## Top debtor exposure

| Debtor | Exposure | Share |
|---|---:|---:|
""" + ("\n".join([
    f"| {d['name']} | {d['exposure']:,.2f} | {d['share'] * 100:.1f}% |"
    for d in metrics["top_debtors"]
]) if metrics["top_debtors"] else "| None | 0 | 0% |") + f"""

## Highest-risk invoices

| Invoice | Debtor | Outstanding | Validity | Risk | Recommendation | Status |
|---|---|---:|---:|---:|---|---|
""" + "\n".join([
    f"| {r['invoice_number']} | {r['debtor_name']} | {r['outstanding']:,.2f} | {r['validity_score']:.1f} | {r['risk_score']:.1f} | {r['recommended_decision']} | {r['lifecycle_status']} |"
    for _, r in top_risk.iterrows()
]) + """

## Limitations

- This report is generated from synthetic or locally imported data.
- Deterministic rules are explainable but do not prove causality.
- Real-world model performance claims require representative, permissioned, independently evaluated data.
- Financial outputs must be reviewed by authorized humans before financing decisions.
"""

    st.subheader("Preview")
    st.markdown(md)

    c1, c2, c3 = st.columns(3)
    c1.download_button("Download Markdown report", md.encode("utf-8"), f"receivables_risk_report_{metrics['as_of']}.md", "text/markdown")
    c2.download_button("Download scored CSV", scored_export.to_csv(index=False).encode("utf-8"), f"scored_invoices_{metrics['as_of']}.csv", "text/csv")
    c3.download_button("Download metrics JSON", json.dumps(json_ready(metrics), indent=2).encode("utf-8"), f"metrics_{metrics['as_of']}.json", "application/json")


def render_settings() -> None:
    st.header("Audit, Settings, Backup, and Quality Checks")

    user = st.session_state.get("user_name", "Demo User")
    role = st.session_state.get("user_role", "Read-only Stakeholder")

    st.subheader("Audit history")
    with get_conn() as conn:
        audit_df = pd.read_sql_query("SELECT * FROM audit_events ORDER BY created_at DESC LIMIT 200", conn)
    st.dataframe(audit_df, use_container_width=True)

    st.subheader("Backup and restore")
    if has_perm(role, "backup"):
        if os.path.exists(DB_PATH):
            with open(DB_PATH, "rb") as f:
                db_bytes = f.read()
            st.download_button("Download SQLite backup", db_bytes, f"invoice_financing_backup_{utcnow().strftime('%Y%m%d_%H%M%S')}.db", "application/octet-stream")
        else:
            st.info("Database file not found.")
    else:
        st.info("Backup permission required.")

    if has_perm(role, "restore"):
        restore_file = st.file_uploader("Restore SQLite database", type=["db"])
        if restore_file:
            if st.button("Restore database"):
                ok, msg = restore_database(restore_file.getvalue(), user, role)
                if ok:
                    st.success(f"Restored. Pre-restore backup: {msg}")
                    st.rerun()
                else:
                    st.error(f"Restore failed: {msg}")
    else:
        st.info("Restore permission required.")

    st.subheader("Retention")
    retention_days = st.number_input("Retain audit/upload logs for days", min_value=1, max_value=3650, value=365)
    confirm_purge = st.checkbox("I understand purge only removes operational logs/upload metadata, not invoice financial records")
    if has_perm(role, "purge"):
        if st.button("Purge expired operational logs"):
            if not confirm_purge:
                st.error("Confirm purge first.")
            else:
                removed = purge_expired(int(retention_days), user, role)
                st.success(f"Purged {removed['audit']} audit rows and {removed['uploads']} upload records.")
    else:
        st.info("Purge permission required.")

    st.subheader("Built-in quality checks")
    if st.button("Run self tests"):
        with st.spinner("Running tests..."):
            results = run_self_tests()
        st.dataframe(results, use_container_width=True)
        fails = results[results["status"] == "FAIL"]
        if fails.empty:
            st.success("All executed tests passed.")
        else:
            st.error(f"{len(fails)} tests failed.")


def restore_database(data: bytes, actor: str, role: str) -> Tuple[bool, str]:
    ensure_dirs()
    temp = os.path.join(UPLOAD_DIR, f"restore_{uuid.uuid4().hex}.db")
    with open(temp, "wb") as f:
        f.write(data)

    try:
        conn = sqlite3.connect(temp)
        integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
        conn.close()
        if integrity != "ok":
            raise ValueError("SQLite integrity check failed.")

        backup = os.path.join(UPLOAD_DIR, f"pre_restore_{utcnow().strftime('%Y%m%d_%H%M%S')}.db")
        if os.path.exists(DB_PATH):
            shutil.copy2(DB_PATH, backup)

        os.replace(temp, DB_PATH)

        with get_conn() as conn:
            audit(conn, actor, role, "restore", "database", DB_PATH, {"backup": backup})
            conn.commit()

        return True, backup

    except Exception as e:
        try:
            if os.path.exists(temp):
                os.remove(temp)
        except Exception:
            pass
        return False, str(e)


def purge_expired(retention_days: int, actor: str, role: str) -> Dict[str, int]:
    cutoff = (utcnow() - timedelta(days=retention_days)).isoformat()
    removed_uploads = 0
    removed_audit = 0

    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, stored_path, error_report_path FROM uploads WHERE uploaded_at < ?",
            (cutoff,),
        ).fetchall()
        for r in rows:
            for p in [r["stored_path"], r["error_report_path"]]:
                if p and os.path.exists(p):
                    try:
                        os.remove(p)
                    except Exception:
                        pass
            removed_uploads += 1

        conn.execute("DELETE FROM uploads WHERE uploaded_at < ?", (cutoff,))
        cur = conn.execute("DELETE FROM audit_events WHERE created_at < ?", (cutoff,))
        removed_audit = cur.rowcount or 0

        audit(conn, actor, role, "purge", "operational_logs", "retention",
              {"retention_days": retention_days, "uploads": removed_uploads, "audit": removed_audit})
        conn.commit()

    return {"uploads": removed_uploads, "audit": removed_audit}


def run_self_tests() -> pd.DataFrame:
    results: List[Dict[str, str]] = []

    def check(name: str, condition: bool, detail: str = "") -> None:
        results.append({"test": name, "status": "PASS" if condition else "FAIL", "detail": detail})

    as_of = pd.Timestamp("2026-01-31")
    base_inv = {
        "id": "test-1",
        "invoice_number": "TEST-001",
        "debtor_id": "D-TEST",
        "issue_date": pd.Timestamp("2026-01-01"),
        "due_date": pd.Timestamp("2026-01-16"),
        "amount": 1000.0,
        "currency": "USD",
        "description": "Test",
    }
    debtor = {"id": "D-TEST", "name": "Test Debtor", "credit_limit": 10000.0}

    res = compute_invoice_assessment(base_inv, debtor, [], as_of, 1)
    check("valid baseline score is 100", abs(res["validity_score"] - 100.0) < 0.01, str(res["validity_score"]))
    check("valid baseline recommended approve", res["recommended_decision"] == "Approve", res["recommended_decision"])

    res_missing = compute_invoice_assessment(base_inv, None, [], as_of, 1)
    check("missing debtor deducts 40", abs(res_missing["validity_score"] - 60.0) < 0.01, str(res_missing["validity_score"]))

    neg = dict(base_inv)
    neg["amount"] = -1.0
    res_neg = compute_invoice_assessment(neg, debtor, [], as_of, 1)
    check("negative amount deducts 40", abs(res_neg["validity_score"] - 60.0) < 0.01, str(res_neg["validity_score"]))

    future = dict(base_inv)
    future["issue_date"] = pd.Timestamp("2026-02-01")
    res_future = compute_invoice_assessment(future, debtor, [], as_of, 1)
    check("future issue deducts 25", abs(res_future["validity_score"] - 75.0) < 0.01, str(res_future["validity_score"]))

    due_before = dict(base_inv)
    due_before["due_date"] = pd.Timestamp("2025-12-31")
    res_due = compute_invoice_assessment(due_before, debtor, [], as_of, 1)
    check("due before issue deducts 20", abs(res_due["validity_score"] - 80.0) < 0.01, str(res_due["validity_score"]))

    overpay = [{"payment_date": pd.Timestamp("2026-01-20"), "amount": 1200.0}]
    res_over = compute_invoice_assessment(base_inv, debtor, overpay, as_of, 1)
    check("overpayment deducts 25", abs(res_over["validity_score"] - 75.0) < 0.01, str(res_over["validity_score"]))

    stale = dict(base_inv)
    stale["due_date"] = pd.Timestamp("2025-09-01")
    res_stale = compute_invoice_assessment(stale, debtor, [], as_of, 1)
    check("stale unpaid bucket 90+", res_stale["aging_bucket"] == "90+", res_stale["aging_bucket"])

    exposure = pd.Series([50.0, 50.0], index=["A", "B"])
    conc = calculate_concentration(exposure, 100.0, {"A": {"name": "A"}, "B": {"name": "B"}})
    check("concentration HHI 50/50 is 0.5", abs(conc["hhi"] - 0.5) < 0.001, str(conc["hhi"]))

    bad_df = pd.DataFrame([{
        "invoice_number": "BAD-TEST",
        "debtor_id": "D-1001",
        "issue_date": "2026-01-01",
        "due_date": "2026-01-31",
        "amount": -10,
        "currency": "USD",
        "description": "bad",
    }])
    with get_conn() as conn:
        _, issues = validate_import(bad_df, "invoices", conn)
    blocking = issues[issues["severity"] == "error"] if not issues.empty else issues
    check("import validation rejects negative amount", not blocking.empty, f"errors={len(blocking)}")

    metrics, _ = calculate_portfolio_metrics(utcnow().date())
    check("portfolio metrics returns non-negative outstanding", metrics["total_outstanding"] >= 0, str(metrics["total_outstanding"]))

    if cv2 is not None:
        vb = st.session_state.get("demo_video_bytes") or generate_demo_video_bytes()
        check("demo video generation produces bytes", vb is not None and len(vb) > 1000, f"bytes={0 if vb is None else len(vb)}")
    else:
        results.append({"test": "demo video generation", "status": "SKIP", "detail": "OpenCV unavailable"})

    return pd.DataFrame(results)


def render_docs() -> None:
    st.header("Product & Governance Docs")

    with st.expander("One-page product brief", expanded=True):
        st.markdown(
            """
            **Product:** Invoice Financing and Receivables Risk Intelligence  
            **Mission:** Assess invoice validity signals, debtor concentration, payment delays, and receivables aging to support human-reviewed financing decisions.  
            **Initial customer segment:** Treasury teams, finance operations, lenders, fintech providers, and financial institutions needing a local, explainable decision-support workflow.  
            **Narrow MVP workflow:** Import sample or customer-provided CSV/XLSX invoice, payment, and debtor data → validate rows → calculate deterministic risk and aging → explain each output → reviewer approves, rejects, corrects, or overrides with reason → export report and audit trail.  
            **Primary value:** Reduce manual spreadsheet review time, make exceptions visible, preserve evidence, and enforce human approval boundaries.  
            **Explicit limits:** Synthetic/demo data only in this build. Not regulated financial advice. No autonomous lending. No promised returns. Real customer impact must be validated through pilots with domain experts.
            """
        )

    with st.expander("User roles and permissions"):
        perm_rows = []
        for role in ROLES:
            perm_rows.append({"Role": role, "Permissions": ", ".join(sorted(PERMISSIONS[role]))})
        st.dataframe(pd.DataFrame(perm_rows), use_container_width=True)

    with st.expander("Central workflow and exception path"):
        st.code(
            """
Actor flow:
Source file / manual entry
  -> Import preview
  -> Row validation
      -> If blocking errors: reject file, download actionable error report, preserve rejected upload for audit
      -> If warnings only: allow import with warnings recorded
  -> Validated record created
  -> Deterministic domain engine calculates validity, aging, concentration, delay, risk
  -> Exception path:
      missing/contradictory/late/out-of-range data -> lifecycle_status = exception
      -> analyst corrects or re-imports
      -> reviewer marks corrected with reason
  -> Analyst submits for review
  -> Reviewer approves / rejects / overrides with reason
  -> Decision and audit event stored
  -> Export report / archive
            """,
            language="text",
        )

    with st.expander("Record lifecycle"):
        lifecycle = pd.DataFrame([
            {"Stage": "Created", "Description": "Imported or seeded record receives stable ID, timestamps, provenance, schema version."},
            {"Stage": "Validated", "Description": "Passed blocking validation; ready for scoring."},
            {"Stage": "Exception", "Description": "Missing, contradictory, late, duplicate, or out-of-range data."},
            {"Stage": "Under review", "Description": "Analyst/operator submitted for human review."},
            {"Stage": "Corrected", "Description": "Reviewer or admin marked data issue corrected with reason."},
            {"Stage": "Approved", "Description": "Authorized reviewer approved financing recommendation."},
            {"Stage": "Rejected", "Description": "Authorized reviewer rejected financing request."},
            {"Stage": "Archived", "Description": "Closed record retained for audit and export."},
        ])
        st.dataframe(lifecycle, use_container_width=True)

    with st.expander("Prioritized backlog"):
        backlog = pd.DataFrame([
            {"Release": "MVP", "Capability": "Local SQLite, synthetic demo data, CSV/XLSX import, validation, deterministic risk/aging/concentration, Streamlit review workflow, exports, audit, self-tests, generated demo video."},
            {"Release": "Post-MVP", "Capability": "Multi-user PostgreSQL, real authentication, SSO, correction workflows, document attachments, reference-data versioning, scheduled imports, API integration adapters."},
            {"Release": "Enterprise", "Capability": "Tenant isolation, advanced RBAC, audit exports, deployment controls, monitoring, SLA documentation, model governance, licensed external data connectors, regulated deployment review."},
        ])
        st.dataframe(backlog, use_container_width=True)

    with st.expander("Measurable acceptance checklist"):
        checklist = pd.DataFrame([
            {"Acceptance item": "New developer can install and run on Windows from README commands.", "Pass condition": "streamlit run app.py opens Live Demo without manual code edits.", "Fail condition": "Import error, missing dependency, or blank dashboard."},
            {"Acceptance item": "Documented sample file imports successfully.", "Pass condition": "Valid invoices sample imports with row count and audit event.", "Fail condition": "No records created or silent failure."},
            {"Acceptance item": "Malformed rows are rejected with actionable errors.", "Pass condition": "Invalid sample shows row/field/issue/value/suggestion and blocks import.", "Fail condition": "Bad rows inserted or vague error."},
            {"Acceptance item": "Calculations are reproducible.", "Pass condition": "Same as-of date and data produce same metrics and scores.", "Fail condition": "Results vary without version change."},
            {"Acceptance item": "Outputs are traceable.", "Pass condition": "Explanation panel shows inputs, flags, contributors, rule version, warnings, reviewer status.", "Fail condition": "Score shown without evidence."},
            {"Acceptance item": "Report export works.", "Pass condition": "Markdown, CSV, and JSON downloads generated.", "Fail condition": "Export button missing or file empty."},
            {"Acceptance item": "Automated tests run.", "Pass condition": "Built-in self tests show PASS/FAIL/SKIP from actual execution.", "Fail condition": "Tests fabricated or not executable."},
            {"Acceptance item": "Live demo starts automatically.", "Pass condition": "Synthetic data loads, simulation runs, MP4 video generated/displayed.", "Fail condition": "Blank dashboard or static mockup only."},
        ])
        st.dataframe(checklist, use_container_width=True)

    with st.expander("Risk register"):
        risks = pd.DataFrame([
            {"Risk": "Domain validity", "Description": "Rules may not reflect actual lender policy.", "Mitigation": "Qualified practitioner review, configurable thresholds post-MVP, human approval required."},
            {"Risk": "Data quality", "Description": "Missing, duplicate, contradictory, or late data can mislead.", "Mitigation": "Row-level validation, exception lifecycle, error reports, lineage, audit."},
            {"Risk": "Security", "Description": "Local app may be mistaken for production auth system.", "Mitigation": "Role simulation only, no secrets, server-side permission checks in code, production requires real auth/SSO."},
            {"Risk": "Adoption", "Description": "Finance teams may not trust opaque scores.", "Mitigation": "Deterministic formulas, explanation panel, override reason, exportable evidence."},
            {"Risk": "Operational failure", "Description": "Corrupt import or database lock can disrupt workflow.", "Mitigation": "Preserve originals, reject bad rows, backup/restore, integrity check, transactional inserts."},
            {"Risk": "Regulatory misrepresentation", "Description": "Prototype could be interpreted as regulated financial advice.", "Mitigation": "Explicit disclaimers, no autonomous lending, human approval boundary, pilot governance."},
        ])
        st.dataframe(risks, use_container_width=True)

    with st.expander("Data dictionary (representative fields)"):
        dd = pd.DataFrame([
            {"Entity": "invoices", "Field": "id", "Type": "TEXT", "Meaning": "Stable record identifier", "Validation": "UUID"},
            {"Entity": "invoices", "Field": "invoice_number", "Type": "TEXT", "Meaning": "Business invoice reference", "Validation": "Non-empty; unique with debtor_id"},
            {"Entity": "invoices", "Field": "debtor_id", "Type": "TEXT", "Meaning": "Debtor organization reference", "Validation": "Must exist in organizations"},
            {"Entity": "invoices", "Field": "issue_date", "Type": "TEXT date", "Meaning": "Invoice issue date", "Validation": "ISO date; not future on import"},
            {"Entity": "invoices", "Field": "due_date", "Type": "TEXT date", "Meaning": "Payment due date", "Validation": "ISO date; not before issue"},
            {"Entity": "invoices", "Field": "amount", "Type": "REAL", "Unit": "Currency units", "Meaning": "Invoice gross amount", "Validation": "Positive; <= configured max"},
            {"Entity": "invoices", "Field": "currency", "Type": "TEXT", "Meaning": "ISO currency code", "Validation": "Allowed reference catalog"},
            {"Entity": "payments", "Field": "payment_date", "Type": "TEXT date", "Meaning": "Cash receipt date", "Validation": "ISO date; not future; not before issue"},
            {"Entity": "payments", "Field": "amount", "Type": "REAL", "Unit": "Currency units", "Meaning": "Payment amount", "Validation": "Positive; cumulative <= invoice tolerance"},
            {"Entity": "organizations", "Field": "credit_limit", "Type": "REAL", "Unit": "Currency units", "Meaning": "Internal exposure limit", "Validation": "Zero or positive"},
            {"Entity": "audit_events", "Field": "actor", "Type": "TEXT", "Meaning": "User or system", "Validation": "Required"},
            {"Entity": "uploads", "Field": "checksum", "Type": "TEXT", "Meaning": "SHA-256 of original file", "Validation": "Required for lineage"},
        ])
        st.dataframe(dd, use_container_width=True)

    with st.expander("Security, threat model, and sensitive-data rules"):
        st.markdown(
            """
            - This MVP uses local role selection for demonstration; production must use authenticated identity and server-enforced RBAC.
            - Password hashing is not implemented because no real authentication is enabled. If added, use a recognized password-hashing library.
            - SQL uses parameterized statements.
            - File uploads are limited to CSV/XLSX/DB where applicable and stored outside relational rows with checksums.
            - Sensitive account/personal data should not be placed in this local demo without governance approval.
            - Audit events record material changes, approvals, overrides, imports, exports, restores, and purges.
            - Secrets belong in environment variables or a secrets manager in deployment, not source control.
            - HTTPS, network ACLs, backup encryption, and retention policy are deployment responsibilities for team/enterprise editions.
            """
        )

    with st.expander("Backup, restore, and PostgreSQL migration plan"):
        st.markdown(
            """
            **Backup:** Download the SQLite file from Audit & Settings. Store outside the app directory with checksum.  
            **Restore:** Upload a SQLite `.db` file; app runs `PRAGMA integrity_check`, creates pre-restore backup, then replaces database.  
            **PostgreSQL migration path:**
            1. Keep schema definitions in migration scripts.
            2. Export SQLite tables to CSV or use `sqlalchemy`/`psycopg` ETL.
            3. Create PostgreSQL schema with constraints, indexes, and sequences where appropriate.
            4. Add real authentication, RBAC, row-level security if multi-tenant.
            5. Move large files to object storage or managed volume; keep metadata/checksums in PostgreSQL.
            6. Validate row counts, checksums, referential integrity, and report reproducibility.
            7. Run parallel read-only comparison before cutover.
            """
        )

    with st.expander("License inventory and commercial-use note"):
        lic = pd.DataFrame([
            {"Dependency": "streamlit", "License": "BSD-3-Clause"},
            {"Dependency": "pandas", "License": "BSD-3-Clause"},
            {"Dependency": "numpy", "License": "BSD-3-Clause / similar permissive"},
            {"Dependency": "plotly", "License": "MIT"},
            {"Dependency": "opencv-python-headless", "License": "Apache-2.0"},
            {"Dependency": "openpyxl", "License": "MIT"},
        ])
        st.dataframe(lic, use_container_width=True)
        st.caption("Review licenses before redistribution or commercial packaging. This app uses no paid API keys.")


def render_page(page: str) -> None:
    if page == "Live Demo":
        render_live_demo()
    elif page == "Data Import":
        render_import()
    elif page == "Records":
        render_records()
    elif page == "Risk Analysis":
        render_analysis()
    elif page == "Review & Approvals":
        render_review()
    elif page == "Reports":
        render_reports()
    elif page == "Audit & Settings":
        render_settings()
    elif page == "Product & Governance Docs":
        render_docs()
    else:
        render_live_demo()


def main() -> None:
    with st.spinner("Preparing local synthetic demo database..."):
        init_schema()
        seed_if_empty()

    with st.sidebar:
        st.header("Access")
        default_user = st.session_state.get("user_name", "Demo User")
        user = st.text_input("User name", default_user)
        stored_role = st.session_state.get("user_role", "Reviewer")
        role_index = ROLES.index(stored_role) if stored_role in ROLES else 2
        role = st.selectbox("Role", ROLES, index=role_index)
        st.session_state.user_name = user
        st.session_state.user_role = role

        st.caption("Local synthetic demo. Human approval required. Not regulated financial advice.")

        page = st.radio(
            "Navigate",
            [
                "Live Demo",
                "Data Import",
                "Records",
                "Risk Analysis",
                "Review & Approvals",
                "Reports",
                "Audit & Settings",
                "Product & Governance Docs",
            ],
        )

    render_page(page)


main()