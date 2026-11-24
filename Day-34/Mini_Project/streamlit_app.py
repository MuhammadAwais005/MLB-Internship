"""
Streamlit front end for the Student Management REST API.

Streamlit Cloud can only run Streamlit apps, so this file starts the FastAPI
server (app/main.py) in a background thread and talks to it over HTTP with
`requests`, exactly like Postman would.

Run locally:
    streamlit run streamlit_app.py

To use an API that is already running somewhere else, set the API_URL
environment variable (e.g. API_URL=http://127.0.0.1:8000) and the built-in
server will not be started.
"""

import os
import sys
import threading
import time
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

API_PORT = int(os.getenv("API_PORT", "8000"))
EXTERNAL_API = os.getenv("API_URL")
API_URL = (EXTERNAL_API or f"http://127.0.0.1:{API_PORT}").rstrip("/")

st.set_page_config(page_title="Student Management", page_icon="🎓", layout="wide")


# ---------------------------------------------------------------------------
# Start the FastAPI server once per Streamlit process
# ---------------------------------------------------------------------------
def _api_is_up() -> bool:
    try:
        return requests.get(f"{API_URL}/health", timeout=1).ok
    except requests.RequestException:
        return False


@st.cache_resource(show_spinner="Starting the API server...")
def start_api_server() -> bool:
    if EXTERNAL_API or _api_is_up():
        return True

    import uvicorn

    from app.main import app

    config = uvicorn.Config(app, host="127.0.0.1", port=API_PORT, log_level="warning")
    server = uvicorn.Server(config)
    threading.Thread(target=server.run, daemon=True).start()

    for _ in range(50):  # wait up to ~10 seconds
        if _api_is_up():
            return True
        time.sleep(0.2)
    return False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def call_api(method: str, path: str, **kwargs):
    """Send a request and return (ok, json_body)."""
    try:
        res = requests.request(method, f"{API_URL}{path}", timeout=10, **kwargs)
        return res.ok, res.json()
    except requests.RequestException as exc:
        return False, {"error": f"Could not reach the API: {exc}"}


def show_error(body: dict):
    st.error(f"❌ {body.get('error', 'Something went wrong')}")
    for d in body.get("details") or []:
        st.caption(f"• **{d['field']}**: {d['message']}")


def students_table(students: list[dict]):
    if not students:
        st.info("No students found.")
        return
    df = pd.DataFrame(students)[["id", "name", "age", "email", "course", "gpa"]]
    st.dataframe(df, hide_index=True)


def student_form(key: str, defaults: dict | None = None, require_all: bool = True):
    """Render input fields and return a dict of the values entered."""
    d = defaults or {}
    c1, c2 = st.columns(2)
    name = c1.text_input("Name", value=d.get("name", ""), key=f"{key}_name")
    email = c2.text_input("Email", value=d.get("email", ""), key=f"{key}_email")
    course = c1.text_input("Course", value=d.get("course", ""), key=f"{key}_course")
    age = c2.number_input("Age", min_value=0, max_value=120, value=int(d.get("age", 18)), step=1, key=f"{key}_age")
    gpa_text = c1.text_input(
        "GPA (optional, 0–4)",
        value="" if d.get("gpa") is None else str(d["gpa"]),
        key=f"{key}_gpa",
    )
    data = {"name": name, "email": email, "course": course, "age": int(age)}
    if gpa_text.strip():
        try:
            data["gpa"] = float(gpa_text)
        except ValueError:
            data["gpa"] = gpa_text  # let the API reject it with a proper message
    elif not require_all:
        data["gpa"] = None
    return data


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
st.title("🎓 Student Management System")
st.caption("A Streamlit UI on top of a FastAPI REST API, with data kept in memory.")

if not start_api_server():
    st.error(f"The API server at {API_URL} could not be started.")
    st.stop()

tab_view, tab_add, tab_search, tab_update, tab_delete = st.tabs(
    ["📋 View all", "➕ Add", "🔍 Search", "✏️ Update", "🗑️ Delete"]
)

# View all --------------------------------------------------------------------
with tab_view:
    st.subheader("All students")
    ok, body = call_api("GET", "/students")
    if ok:
        st.write(f"**{body['count']}** student(s)")
        students_table(body["students"])
    else:
        show_error(body)

    st.divider()
    st.subheader("Find by id")
    sid = st.number_input("Student id", min_value=1, step=1, key="view_id")
    if st.button("Get student", key="view_btn"):
        ok, body = call_api("GET", f"/students/{int(sid)}")
        if ok:
            st.json(body)
        else:
            show_error(body)

# Add -------------------------------------------------------------------------
with tab_add:
    st.subheader("Add a new student")
    with st.form("add_form", clear_on_submit=False):
        data = student_form("add")
        submitted = st.form_submit_button("Add student", type="primary")
    if submitted:
        payload = {k: v for k, v in data.items() if v is not None}
        ok, body = call_api("POST", "/students", json=payload)
        if ok:
            st.success(f"✅ Added **{body['name']}** with id {body['id']}")
            st.json(body)
        else:
            show_error(body)

# Search ----------------------------------------------------------------------
with tab_search:
    st.subheader("Search students")
    c1, c2, c3 = st.columns(3)
    q_name = c1.text_input("Name contains")
    q_course = c2.text_input("Course contains")
    q_email = c3.text_input("Exact email")
    c4, c5 = st.columns(2)
    q_min = c4.number_input("Min age (0 = any)", min_value=0, step=1)
    q_max = c5.number_input("Max age (0 = any)", min_value=0, step=1)
    if st.button("Search", type="primary"):
        params = {"name": q_name.strip(), "course": q_course.strip(), "email": q_email.strip()}
        params = {k: v for k, v in params.items() if v}
        if q_min:
            params["min_age"] = int(q_min)
        if q_max:
            params["max_age"] = int(q_max)
        ok, body = call_api("GET", "/students/search", params=params)
        if ok:
            st.write(f"**{body['count']}** match(es)")
            students_table(body["students"])
        else:
            show_error(body)

# Update ----------------------------------------------------------------------
with tab_update:
    st.subheader("Update student information")
    uid = st.number_input("Id of the student to update", min_value=1, step=1, key="upd_id")
    ok, current = call_api("GET", f"/students/{int(uid)}")
    if not ok:
        show_error(current)
    else:
        st.caption(f"Editing **{current['name']}**. Only the fields you change are sent (PATCH).")
        with st.form("update_form"):
            data = student_form(f"upd{int(uid)}", defaults=current, require_all=False)
            submitted = st.form_submit_button("Save changes", type="primary")
        if submitted:
            changes = {k: v for k, v in data.items() if v != current.get(k)}
            if not changes:
                st.info("Nothing changed.")
            else:
                ok, body = call_api("PATCH", f"/students/{int(uid)}", json=changes)
                if ok:
                    st.success(f"✅ {body['message']}")
                    st.json(body["student"])
                else:
                    show_error(body)

# Delete ----------------------------------------------------------------------
with tab_delete:
    st.subheader("Delete a student")
    did = st.number_input("Id of the student to delete", min_value=1, step=1, key="del_id")
    ok, target = call_api("GET", f"/students/{int(did)}")
    if ok:
        st.warning(f"You are about to delete **{target['name']}** ({target['email']}).")
        confirm = st.checkbox("Yes, I want to delete this student", key=f"confirm_{int(did)}")
        if st.button("Delete", type="primary", disabled=not confirm):
            ok, body = call_api("DELETE", f"/students/{int(did)}")
            if ok:
                st.success(f"✅ {body['message']}")
            else:
                show_error(body)
    else:
        show_error(target)

# Sidebar (rendered last so the count reflects any change made above)
with st.sidebar:
    st.header("API status")
    ok, health = call_api("GET", "/health")
    if ok:
        st.success("API is healthy ✅")
        st.metric("Total students", health["total_students"])
    else:
        show_error(health)
    st.caption(f"API URL: `{API_URL}`")
    st.caption("Data resets whenever the app restarts.")
