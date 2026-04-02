import json
import os
from datetime import datetime

SHEET_ID = "1Qfrt7afqBYDiv5THOtBvHCC-puomjWK_Dm1MpFWTzJU"
SHEET_NAME = "Sheet1"

def get_client():
    try:
        import gspread
        from google.oauth2.service_account import Credentials

        scopes = [
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/drive"
        ]

        # Try Streamlit secrets first (cloud)
        try:
            import streamlit as st
            creds_dict = dict(st.secrets["gcp_service_account"])
            creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
        except Exception:
            # Fall back to local file
            creds_path = os.path.join(os.path.dirname(__file__), "gcp_credentials.json")
            creds = Credentials.from_service_account_file(creds_path, scopes=scopes)

        return gspread.authorize(creds)
    except Exception as e:
        print("[Sheets] Client init failed:", str(e)[:80])
        return None

def ensure_headers(sheet):
    try:
        first_row = sheet.row_values(1)
        if not first_row:
            sheet.append_row([
                "Timestamp", "Query", "Intent", "Pipeline Type",
                "Critic Score", "Latency (s)", "Cache Status",
                "Agents Used", "Feedback Loop", "Hallucination Flagged",
                "Run ID", "Quality Label"
            ])
    except Exception:
        pass

def get_score_label(score):
    if score >= 8:
        return "Excellent"
    elif score >= 7:
        return "Strong"
    elif score >= 5:
        return "Good"
    elif score > 0:
        return "Fair"
    return "—"

def log_run_to_sheets(run: dict) -> bool:
    try:
        client = get_client()
        if not client:
            return False

        spreadsheet = client.open_by_key(SHEET_ID)
        sheet = spreadsheet.worksheet(SHEET_NAME)
        ensure_headers(sheet)

        score = run.get("critic_score", -1)
        agents = run.get("agents_used", [])

        sheet.append_row([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            run.get("query", "")[:100],
            run.get("intent", ""),
            run.get("pipeline_type", ""),
            score if score > 0 else "—",
            round(run.get("total_latency", 0), 1),
            run.get("cache_status", ""),
            ", ".join(agents) if agents else "",
            "Yes" if run.get("feedback_loop_used") else "No",
            "Yes" if run.get("hallucination_flagged") else "No",
            run.get("run_id", ""),
            get_score_label(score)
        ])
        print("[Sheets] Run logged successfully")
        return True
    except Exception as e:
        print("[Sheets] Log failed:", str(e)[:80])
        return False

def read_all_runs() -> list:
    try:
        client = get_client()
        if not client:
            return []
        spreadsheet = client.open_by_key(SHEET_ID)
        sheet = spreadsheet.worksheet(SHEET_NAME)
        rows = sheet.get_all_records()
        return rows
    except Exception as e:
        print("[Sheets] Read failed:", str(e)[:80])
        return []

def get_sheets_stats() -> dict:
    rows = read_all_runs()
    if not rows:
        return {"total_runs": 0}

    scores = [int(r["Critic Score"]) for r in rows
              if str(r.get("Critic Score","")).isdigit()]
    latencies = [float(r["Latency (s)"]) for r in rows
                 if str(r.get("Latency (s)","")).replace(".","").isdigit()]
    cache_hits = sum(1 for r in rows if r.get("Cache Status") == "hit")
    fb_loops   = sum(1 for r in rows if r.get("Feedback Loop") == "Yes")
    hall_flags = sum(1 for r in rows if r.get("Hallucination Flagged") == "Yes")

    total = len(rows)
    return {
        "total_runs":        total,
        "avg_critic_score":  round(sum(scores)/len(scores),1) if scores else None,
        "median_score":      sorted(scores)[len(scores)//2] if scores else None,
        "avg_latency":       round(sum(latencies)/len(latencies),1) if latencies else None,
        "cache_hit_rate":    round(cache_hits/total*100,1),
        "feedback_loop_rate":round(fb_loops/total*100,1),
        "hallucination_rate":round(hall_flags/total*100,1),
        "recent_runs":       rows[-10:][::-1]
    }

if __name__ == "__main__":
    stats = get_sheets_stats()
    print("Total runs:", stats.get("total_runs"))
    print("Avg score:", stats.get("avg_critic_score"))
