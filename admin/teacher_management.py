"""Admin: manage teacher accounts."""
import streamlit as st
from database.database import query, execute
from authentication.auth_service import log_activity


def render_teacher_management():
    st.markdown("## 👥 Teacher Management")

    # filters
    col1, col2 = st.columns(2)
    with col1:
        status_filter = st.selectbox(
            "Filter by status",
            ["All", "pending", "approved", "rejected", "inactive"],
        )
    with col2:
        search = st.text_input("Search by name or username")

    sql = "SELECT * FROM users WHERE role = 'teacher'"
    params = []
    if status_filter != "All":
        sql += " AND approval_status = ?"
        params.append(status_filter)
    sql += " ORDER BY created_at DESC"

    teachers = query(sql, tuple(params))

    if search:
        s = search.lower()
        teachers = [t for t in teachers
                    if s in t["full_name"].lower() or s in t["username"].lower()]

    if not teachers:
        st.info("No teachers found.")
        return

    st.write(f"**{len(teachers)} teacher(s)**")

    for t in teachers:
        with st.expander(f"{t['full_name']} — {t['username']} ({t['approval_status']})"):
            st.write(f"**Email:** {t['email']}")
            st.write(f"**Department:** {t['department'] or '—'}")
            st.write(f"**Subject:** {t['subject'] or '—'}")
            st.write(f"**Registered:** {t['created_at']}")
            st.write(f"**Status:** {t['approval_status']}")

            # UPDATED: 4 columns for buttons
            c1, c2, c3, c4 = st.columns(4)
            
            with c1:
                if t["approval_status"] != "approved":
                    if st.button(f"✅ Approve", key=f"appr_{t['id']}"):
                        execute("UPDATE users SET approval_status='approved' WHERE id=?", (t["id"],))
                        log_activity(st.session_state.user["id"], "approve_teacher", f"Approved {t['username']}")
                        st.success("Approved.")
                        st.rerun()
            with c2:
                if t["approval_status"] not in ("rejected", "approved"):
                    if st.button(f"❌ Reject", key=f"rej_{t['id']}"):
                        execute("UPDATE users SET approval_status='rejected' WHERE id=?", (t["id"],))
                        log_activity(st.session_state.user["id"], "reject_teacher", f"Rejected {t['username']}")
                        st.warning("Rejected.")
                        st.rerun()
            with c3:
                if t["approval_status"] == "approved":
                    if st.button(f"🔒 Deactivate", key=f"deact_{t['id']}"):
                        execute("UPDATE users SET approval_status='inactive' WHERE id=?", (t["id"],))
                        log_activity(st.session_state.user["id"], "deactivate_teacher", f"Deactivated {t['username']}")
                        st.warning("Deactivated.")
                        st.rerun()
                elif t["approval_status"] == "inactive":
                    if st.button(f"🔓 Reactivate", key=f"react_{t['id']}"):
                        execute("UPDATE users SET approval_status='approved' WHERE id=?", (t["id"],))
                        log_activity(st.session_state.user["id"], "reactivate_teacher", f"Reactivated {t['username']}")
                        st.success("Reactivated.")
                        st.rerun()
            with c4:
                # NEW: Delete Button
                if st.button(f"🗑️ Delete", key=f"del_{t['id']}"):
                    # Unassign students first to prevent foreign key errors
                    execute("UPDATE students SET assigned_teacher_id = NULL WHERE assigned_teacher_id = ?", (t["id"],))
                    # Delete the teacher
                    execute("DELETE FROM users WHERE id=?", (t["id"],))
                    log_activity(st.session_state.user["id"], "delete_teacher", f"Deleted {t['username']}")
                    st.success(f"Teacher {t['full_name']} deleted.")
                    st.rerun()