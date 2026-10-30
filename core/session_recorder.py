"""
core/session_recorder.py

User Interaction & Session Recording Engine for cGMP-QC-CoPilot.
Captures user events, parameter shifts, and agent outputs into a tamper-evident timeline,
with browser-based HTML5 screen & audio recording capabilities.

Complies strictly with 21 CFR Part 11 ALCOA+ principles and Guardrail G2 draft standards.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import streamlit as st
import streamlit.components.v1 as components

from core.stamps import DRAFT_STAMP


@dataclass
class InteractionEvent:
    """Represents a single user or system telemetry event in the audit journal."""
    timestamp_iso: str
    event_type: str
    workspace: str
    widget_key: str
    old_val: Any
    new_val: Any
    latency_ms: float
    agent3_status: str
    audit_hash: str
    event_index: int = 1
    prev_hash: str = "GENESIS_BLOCK_0000000000000000000000000000000000000000000000000000"

    @property
    def sha256_hash(self) -> str:
        return self.audit_hash

    @property
    def old_value(self) -> Any:
        return self.old_val

    @property
    def new_value(self) -> Any:
        return self.new_val


class SessionRecorder:
    """Manages active session interaction journal recording and audit exports."""

    def __init__(self) -> None:
        self.is_recording: bool = False
        self.events: List[Dict[str, Any]] = []
        self.start_timestamp: Optional[str] = None
        self.start_perf: Optional[float] = None

    def start(self) -> None:
        """Starts a new recording session or resumes current journal."""
        self.is_recording = True
        if self.start_perf is None:
            self.start_perf = time.perf_counter()
            self.start_timestamp = datetime.now(timezone.utc).isoformat()

    def stop(self) -> None:
        """Stops active recording session."""
        self.is_recording = False

    def reset(self) -> None:
        """Clears all events and resets session clock."""
        self.is_recording = False
        self.events.clear()
        self.start_perf = None
        self.start_timestamp = None

    def record_event(
        self,
        event_type: str,
        workspace: str = "",
        widget_key: str = "",
        old_val: Any = None,
        new_val: Any = None,
        latency_ms: float = 0.0,
        agent3_status: str = "NOMINAL",
        old_value: Any = None,
        new_value: Any = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[InteractionEvent]:
        """Appends a new interaction event to the cryptographic journal."""
        if not self.is_recording:
            return None

        actual_old = old_value if old_value is not None else old_val
        actual_new = new_value if new_value is not None else new_val

        now_iso = datetime.now(timezone.utc).isoformat()
        evt_idx = len(self.events) + 1
        prev_hash = self.events[-1]["audit_hash"] if self.events else "GENESIS_BLOCK_0000000000000000000000000000000000000000000000000000"

        payload = {
            "event_index": evt_idx,
            "timestamp_iso": now_iso,
            "event_type": event_type,
            "workspace": workspace,
            "widget_key": widget_key,
            "old_val": actual_old,
            "new_val": actual_new,
            "latency_ms": round(float(latency_ms), 2),
            "agent3_status": str(agent3_status),
            "prev_hash": prev_hash,
            "metadata": metadata or {},
            "draft_stamp": DRAFT_STAMP,
        }

        payload_bytes = (json.dumps(payload, sort_keys=True, default=str) + prev_hash).encode("utf-8")
        event_hash = hashlib.sha256(payload_bytes).hexdigest()

        event_dict = dict(payload, audit_hash=event_hash)
        self.events.append(event_dict)

        return InteractionEvent(
            timestamp_iso=now_iso,
            event_type=event_type,
            workspace=workspace,
            widget_key=widget_key,
            old_val=actual_old,
            new_val=actual_new,
            latency_ms=round(float(latency_ms), 2),
            agent3_status=str(agent3_status),
            audit_hash=event_hash,
            event_index=evt_idx,
            prev_hash=prev_hash,
        )

    def export_json(self) -> str:
        """Exports the interaction journal as structured JSON."""
        container = {
            "session_summary": {
                "start_timestamp": self.start_timestamp,
                "exported_timestamp": datetime.now(timezone.utc).isoformat(),
                "total_events": len(self.events),
                "is_recording": self.is_recording,
                "disclaimer": DRAFT_STAMP,
            },
            "interaction_journal": self.events,
        }
        return json.dumps(container, indent=2)

    def export_csv(self) -> str:
        """Exports the interaction journal as flat tabular CSV."""
        if not self.events:
            return "event_index,timestamp_utc,event_type,widget_key,old_val,new_val,latency_ms,agent3_status,audit_hash\n"

        output = io.StringIO()
        fieldnames = [
            "event_index",
            "timestamp_utc",
            "event_type",
            "widget_key",
            "old_val",
            "new_val",
            "latency_ms",
            "agent3_status",
            "audit_hash",
        ]
        writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for e in self.events:
            row = {
                "event_index": e.get("event_index", 1),
                "timestamp_utc": e.get("timestamp_iso", ""),
                "event_type": e.get("event_type", ""),
                "widget_key": e.get("widget_key", ""),
                "old_val": str(e.get("old_val", "")),
                "new_val": str(e.get("new_val", "")),
                "latency_ms": e.get("latency_ms", 0.0),
                "agent3_status": e.get("agent3_status", "NOMINAL"),
                "audit_hash": e.get("audit_hash", ""),
            }
            writer.writerow(row)
        return output.getvalue()

    def get_summary(self) -> Dict[str, Any]:
        """Returns summary metrics for session inspection."""
        elapsed_s = (time.perf_counter() - self.start_perf) if (self.start_perf and self.is_recording) else 0.0
        return {
            "is_recording": self.is_recording,
            "event_count": len(self.events),
            "elapsed_seconds": round(elapsed_s, 1),
            "start_timestamp": self.start_timestamp,
        }


def get_session_recorder() -> SessionRecorder:
    """Retrieves or initializes the SessionRecorder from st.session_state."""
    if "session_recorder_instance" not in st.session_state:
        st.session_state["session_recorder_instance"] = SessionRecorder()
    return st.session_state["session_recorder_instance"]


def render_browser_screen_recorder_widget() -> None:
    """
    Renders an in-browser HTML5 Screen & Audio Capture component utilizing
    navigator.mediaDevices.getDisplayMedia to export downloadable .webm clips.
    """
    html_code = """
    <div style="font-family: 'Roboto', 'Google Sans', sans-serif; background: #ffffff; border: 1px solid #e0e3e7; border-radius: 10px; padding: 14px; margin-top: 8px;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
            <div style="display: flex; align-items: center; gap: 8px;">
                <span id="rec-dot" style="display: inline-block; width: 10px; height: 10px; border-radius: 50%; background: #9aa0a6;"></span>
                <strong style="font-size: 13px; color: #202124;">Browser Screen & Audio Media Recorder</strong>
            </div>
            <span id="rec-timer" style="font-size: 12px; font-weight: 700; color: #5f6368; font-variant-numeric: tabular-nums;">00:00</span>
        </div>
        <p style="font-size: 11.5px; color: #5f6368; margin: 0 0 10px 0;">
            Capture full browser tab interactions, sliders, and audio narration directly to an exportable WebM video file:
        </p>
        <div style="display: flex; gap: 8px; flex-wrap: wrap;">
            <button id="btn-start-rec" style="background: #1a73e8; color: #ffffff; border: none; border-radius: 6px; padding: 6px 14px; font-size: 12px; font-weight: 600; cursor: pointer; display: flex; align-items: center; gap: 6px;">
                🎥 Start Screen Capture
            </button>
            <button id="btn-stop-rec" disabled style="background: #e8eaed; color: #5f6368; border: none; border-radius: 6px; padding: 6px 14px; font-size: 12px; font-weight: 600; cursor: not-allowed; display: flex; align-items: center; gap: 6px;">
                ⏹️ Stop & Download WebM
            </button>
        </div>
        <div id="rec-status" style="font-size: 11px; color: #70757a; margin-top: 8px;">
            Status: Ready for capture. Click 'Start Screen Capture' to begin.
        </div>
    </div>

    <script>
    (function() {
        let mediaRecorder = null;
        let recordedChunks = [];
        let timerInterval = null;
        let secondsElapsed = 0;

        const btnStart = document.getElementById('btn-start-rec');
        const btnStop = document.getElementById('btn-stop-rec');
        const recDot = document.getElementById('rec-dot');
        const recTimer = document.getElementById('rec-timer');
        const recStatus = document.getElementById('rec-status');

        function formatTime(s) {
            const mins = Math.floor(s / 60).toString().padStart(2, '0');
            const secs = (s % 60).toString().padStart(2, '0');
            return `${mins}:${secs}`;
        }

        btnStart.addEventListener('click', async () => {
            try {
                const stream = await navigator.mediaDevices.getDisplayMedia({
                    video: { cursor: "always" },
                    audio: true
                });

                recordedChunks = [];
                mediaRecorder = new MediaRecorder(stream, { mimeType: 'video/webm; codecs=vp9' });

                mediaRecorder.ondataavailable = (e) => {
                    if (e.data.size > 0) recordedChunks.push(e.data);
                };

                mediaRecorder.onstop = () => {
                    clearInterval(timerInterval);
                    recDot.style.background = '#9aa0a6';
                    recDot.style.animation = 'none';
                    btnStart.disabled = false;
                    btnStart.style.background = '#1a73e8';
                    btnStart.style.cursor = 'pointer';
                    btnStop.disabled = true;
                    btnStop.style.background = '#e8eaed';
                    btnStop.style.color = '#5f6368';
                    btnStop.style.cursor = 'not-allowed';

                    if (recordedChunks.length > 0) {
                        const blob = new Blob(recordedChunks, { type: 'video/webm' });
                        const url = URL.createObjectURL(blob);
                        const a = document.createElement('a');
                        a.style.display = 'none';
                        a.href = url;
                        a.download = `cGMP-QC-CoPilot_Session_${new Date().toISOString().replace(/[:.]/g, '-')}.webm`;
                        document.body.appendChild(a);
                        a.click();
                        setTimeout(() => {
                            document.body.removeChild(a);
                            window.URL.revokeObjectURL(url);
                        }, 100);
                        recStatus.textContent = 'Capture complete. Download started.';
                    } else {
                        recStatus.textContent = 'Capture finished with zero recorded bytes.';
                    }
                };

                // Stream tracks listener for browser stop-share button
                stream.getVideoTracks()[0].onended = () => {
                    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
                        mediaRecorder.stop();
                    }
                };

                mediaRecorder.start(1000);
                secondsElapsed = 0;
                recTimer.textContent = '00:00';
                timerInterval = setInterval(() => {
                    secondsElapsed++;
                    recTimer.textContent = formatTime(secondsElapsed);
                }, 1000);

                recDot.style.background = '#d93025';
                btnStart.disabled = true;
                btnStart.style.background = '#e8eaed';
                btnStart.style.cursor = 'not-allowed';
                btnStop.disabled = false;
                btnStop.style.background = '#d93025';
                btnStop.style.color = '#ffffff';
                btnStop.style.cursor = 'pointer';
                recStatus.textContent = 'Recording active. Interacting with tabs and controls is being captured.';
            } catch (err) {
                console.error(err);
                recStatus.textContent = 'Capture aborted or permission denied: ' + err.message;
            }
        });

        btnStop.addEventListener('click', () => {
            if (mediaRecorder && mediaRecorder.state !== 'inactive') {
                mediaRecorder.stop();
                mediaRecorder.stream.getTracks().forEach(track => track.stop());
            }
        });
    })();
    </script>
    """
    components.html(html_code, height=140)


def render_session_recorder_modal_or_popover(workspace: str, agent3_status: str) -> None:
    """Renders the top-bar recording controls, state journal, and audit replay."""
    recorder = get_session_recorder()
    summary = recorder.get_summary()

    with st.popover(
        f"⏺️ Session Recorder ({summary['event_count']} Events)",
        help="Capture user interactions, export audit journals, or record browser video clips",
    ):
        st.markdown("#### ⏺️ Interaction Journal & Media Recorder")
        st.caption("21 CFR Part 11 ALCOA+ Attributable Telemetry Recording")

        c_rec1, c_rec2, c_rec3 = st.columns(3)
        with c_rec1:
            if not summary["is_recording"]:
                if st.button("🔴 Start Recording", use_container_width=True, type="primary"):
                    recorder.start()
                    st.rerun()
            else:
                if st.button("⏹️ Stop Recording", use_container_width=True):
                    recorder.stop()
                    st.rerun()

        with c_rec2:
            if st.button("🗑️ Clear Log", use_container_width=True):
                recorder.reset()
                st.rerun()

        with c_rec3:
            st.metric(
                label="Status",
                value="REC ACTIVE" if summary["is_recording"] else "IDLE",
                delta=f"{summary['event_count']} Events",
            )

        # In-Browser Screen Capture HTML5 Widget
        st.markdown("---")
        st.markdown("##### 🎥 Video & Audio Screen Recorder")
        render_browser_screen_recorder_widget()

        # Export Section
        st.markdown("---")
        st.markdown("##### 💾 Export Interaction Journal")
        col_exp1, col_exp2 = st.columns(2)
        with col_exp1:
            json_data = recorder.export_json()
            st.download_button(
                label="⬇️ Download JSON Journal",
                data=json_data,
                file_name=f"cGMP_Interaction_Journal_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True,
            )
        with col_exp2:
            csv_data = recorder.export_csv()
            st.download_button(
                label="⬇️ Download CSV Log",
                data=csv_data,
                file_name=f"cGMP_Interaction_Journal_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True,
            )

        # Audit Replay Mode
        st.markdown("---")
        with st.expander(f"▶️ Audit Replay Mode ({len(recorder.events)} Recorded Steps)", expanded=False):
            if not recorder.events:
                st.info("No recorded events in current session. Click 'Start Recording' and adjust parameters to log telemetry.")
            else:
                st.caption("Step through logged events sequentially to audit biopharmaceutical parameter history:")
                step_idx = st.slider(
                    "Replay Step",
                    min_value=1,
                    max_value=len(recorder.events),
                    value=len(recorder.events),
                    step=1,
                    key="journal_replay_step_slider",
                )
                curr_evt = recorder.events[step_idx - 1]
                st.json(curr_evt)
