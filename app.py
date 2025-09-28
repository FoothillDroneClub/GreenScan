"""Streamlit interface for the Optimism Trash Classification MVP."""

from __future__ import annotations

import io
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import hashlib
import sqlite3

import streamlit as st

from app_utils import (
    ClassificationError,
    DetectionResult,
    MissingDependencyError,
    analyze_image,
    detect_objects,
    load_blip,
    load_classifier,
    load_yolo,
)

APP_DIR = Path(__file__).resolve().parent
DATABASE_FILE = APP_DIR / "users.db"


st.set_page_config(
    page_title="Optimism Trash Classifier",
    page_icon="♻️",
    layout="wide",
)

st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(180deg, #e9f7ef 0%, #f0fff4 45%, #ffffff 100%);
            color: #0b4331;
        }
        .eco-card {
            background: #ffffff;
            border-radius: 18px;
            padding: 1rem 1.75rem 1.5rem;
            border: 1px solid #000000;
            box-shadow: 0 18px 40px rgba(12, 91, 61, 0.14);
        }
        .eco-card h1, .eco-card h3 {
            color: #0b5d3a;
        }
        .main-title {
            font-size: 3.25rem;
            font-weight: 800;
            color: #0b5d3a;
            display: flex;
            align-items: center;
            gap: 0.75rem;
            margin: 0 0 1.75rem;
        }
        .main-title .leaf-icon {
            font-size: 3.5rem;
        }
        .step-heading {
            font-weight: 700;
            color: #0b5d3a;
            margin-top: 0;
            margin-bottom: 0.75rem;
        }
        .stButton > button {
            background: linear-gradient(120deg, #15803d, #22c55e);
            border: none;
            color: #f0fdf4;
            padding: 0.75rem 1.5rem;
            border-radius: 999px;
            font-size: 1.05rem;
            font-weight: 600;
            box-shadow: 0 12px 28px rgba(25, 135, 84, 0.24);
        }
        .stButton > button:hover {
            background: linear-gradient(120deg, #166534, #16a34a);
            box-shadow: 0 18px 32px rgba(25, 135, 84, 0.28);
        }
        .tips-card {
            background: #ffffff;
            color: #0b4331;
            padding: 1.5rem;
            border-radius: 18px;
            border: 1px solid #000000;
        }
        .tips-card ul {
            padding-left: 1.1rem;
        }
        .tips-card li {
            margin-bottom: 0.5rem;
        }
        div[data-testid="stExpander"] {
            background: #ffffff;
            border: 1px solid #000000;
            border-radius: 12px;
        }
        div[data-testid="stForm"] {
            background: #ffffff;
            border: 1px solid #000000;
            border-radius: 12px;
            padding: 1.25rem !important;
            color: #0b4331;
        }
        div[data-testid="stForm"] label,
        div[data-testid="stForm"] legend {
            color: #000000 !important;
        }
        div[data-testid="stForm"] input,
        div[data-testid="stForm"] select,
        div[data-testid="stForm"] textarea {
            background-color: #ffffff !important;
            color: #0b4331 !important;
            border: 1px solid #000000 !important;
            border-radius: 8px !important;
        }
        form#login_form button {
            background: #ffffff !important;
            color: #ffffff !important;
            border: 1px solid #000000 !important;
        }
        form#signup_form button {
            background: #000000 !important;
            color: #ffffff !important;
            border: 1px solid #000000 !important;
        }
        div[role="radiogroup"] label p,
        div[role="radiogroup"] label span {
            color: #000000 !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

RANK_THRESHOLDS: List[Tuple[int, str, str]] = [
    (0, "Newbie Recycler", "🔰"),
    (100, "Intermediate Recycler", "🌱"),
    (200, "Advanced Recycler", "🌿"),
    (500, "Expert Recycler", "🌎"),
    (1000, "Master Recycler", "🏆"),
]

VALID_BADGES = {f"{icon} {label}" for _, label, icon in RANK_THRESHOLDS}


def determine_rank(total_points: int) -> Tuple[str, str]:
    """Return the rank label and icon corresponding to the given point total."""
    for minimum_points, label, icon in reversed(RANK_THRESHOLDS):
        if total_points >= minimum_points:
            return label, icon
    # Fallback to the first threshold if points are somehow negative.
    return RANK_THRESHOLDS[0][1], RANK_THRESHOLDS[0][2]


def normalize_badge(total_points: int, badge: Optional[str]) -> str:
    """Ensure the stored badge matches the current rank definition."""
    expected_badge = get_badge(total_points)
    if not badge or badge not in VALID_BADGES:
        return expected_badge
    if badge != expected_badge:
        return expected_badge
    return badge


DATABASE_SETUP_SQL = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    total_points INTEGER NOT NULL DEFAULT 0,
    last_badge TEXT NOT NULL DEFAULT '🔰 Newbie Recycler'
);

CREATE TABLE IF NOT EXISTS shop_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    item_type TEXT NOT NULL,
    cost INTEGER NOT NULL,
    description TEXT NOT NULL,
    icon TEXT,
    is_active INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS user_shop_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    item_id INTEGER NOT NULL,
    purchased_at TEXT NOT NULL,
    UNIQUE(user_id, item_id),
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (item_id) REFERENCES shop_items(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS user_equipped_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    item_id INTEGER NOT NULL,
    equipped_at TEXT NOT NULL,
    UNIQUE(user_id, item_id),
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (item_id) REFERENCES shop_items(id) ON DELETE CASCADE
);
"""

SESSION_KEY = "active_user"
DEFAULT_AVATAR = "https://i.pravatar.cc/100?img=60"
MASKED_PASSWORD_DISPLAY = "********"


# Seed leaderboard entries (besides the active user).
LEADERBOARD_SEED: List[Dict[str, int]] = [
    {"name": "Ava", "points": 180},
    {"name": "Noah", "points": 120},
    {"name": "Mia", "points": 85},
    {"name": "Leo", "points": 60},
]

SHOP_ITEMS_SEED: List[Dict[str, Any]] = [
    {
        "name": "Eco Hero Title",
        "item_type": "title",
        "cost": 150,
        "description": "Unlock a heroic title that crowns your profile with eco pride.",
        "icon": "🦸",
    },
    {
        "name": "Nature's Guardian Title",
        "item_type": "title",
        "cost": 300,
        "description": "Adopt the Guardian of Green honorific to showcase sustained impact.",
        "icon": "🛡️",
    },
    {
        "name": "Aurora Leaf Particles",
        "item_type": "particles",
        "cost": 200,
        "description": "Surround your rank badge with a dancing aurora of leafy sparks.",
        "icon": "🍃",
    },
    {
        "name": "Starlight Sprout Particles",
        "item_type": "particles",
        "cost": 350,
        "description": "Bathe your badge in starlight sprout particles for extra shine.",
        "icon": "✨",
    },
]


def initialize_state() -> None:
    """Ensure required keys exist in Streamlit session state."""
    st.session_state.setdefault(SESSION_KEY, None)
    st.session_state.setdefault("total_points", 0)
    st.session_state.setdefault("badge", get_badge(st.session_state.total_points))
    st.session_state.setdefault("rank_label", determine_rank(st.session_state.total_points)[0])
    st.session_state.setdefault("rank_icon", determine_rank(st.session_state.total_points)[1])
    st.session_state.setdefault(
        "leaderboard",
        [
            {"name": "You", "points": st.session_state.total_points},
            *LEADERBOARD_SEED,
        ],
    )
    st.session_state.setdefault("last_result", None)
    st.session_state.setdefault("show_auth_form", "login")
    st.session_state.setdefault("auth_error", None)
    st.session_state.setdefault("active_tab", "scanner")
    st.session_state.setdefault("purchased_items", [])
    st.session_state.setdefault("shop_items", [])
    st.session_state.setdefault("shop_item_lookup", {})
    st.session_state.setdefault("shop_initialized", False)
    st.session_state.setdefault("shop_feedback", None)
    st.session_state.setdefault("equipped_item_ids", [])

    if st.session_state[SESSION_KEY]:
        load_active_user_from_db()
    else:
        ensure_shop_seeded()



def get_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(DATABASE_FILE)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON;")
    connection.executescript(DATABASE_SETUP_SQL)
    return connection


def fetch_shop_items() -> List[sqlite3.Row]:
    with get_connection() as connection:
        rows = connection.execute(
            "SELECT id, name, item_type, cost, description, icon, is_active FROM shop_items WHERE is_active = 1 ORDER BY cost ASC",
        ).fetchall()
    return rows


def fetch_user_purchases(user_id: int) -> List[int]:
    with get_connection() as connection:
        rows = connection.execute(
            "SELECT item_id FROM user_shop_items WHERE user_id = ?",
            (user_id,),
        ).fetchall()
    return [row["item_id"] for row in rows]


def refresh_shop_items() -> None:
    shop_items = [
        {
            "id": row["id"],
            "name": row["name"],
            "item_type": row["item_type"],
            "cost": row["cost"],
            "description": row["description"],
            "icon": row["icon"],
        }
        for row in fetch_shop_items()
    ]
    st.session_state.shop_items = shop_items
    st.session_state.shop_item_lookup = {item["id"]: item for item in shop_items}


def purchase_item(item_id: int) -> None:
    active_user = st.session_state.get(SESSION_KEY)
    if not active_user:
        st.session_state.shop_feedback = ("error", "Sign in to purchase items.")
        return

    if item_id in st.session_state.purchased_items:
        st.session_state.shop_feedback = ("info", "You already own this item.")
        return

    item = st.session_state.shop_item_lookup.get(item_id)
    if item is None:
        refresh_shop_items()
        item = st.session_state.shop_item_lookup.get(item_id)
        if item is None:
            st.session_state.shop_feedback = ("error", "Item is no longer available.")
            return

    cost = item["cost"]
    if st.session_state.total_points < cost:
        st.session_state.shop_feedback = ("warning", "You need more eco points to purchase this item.")
        return

    new_points = max(st.session_state.total_points - cost, 0)
    new_badge = get_badge(new_points)
    timestamp = datetime.utcnow().isoformat()

    try:
        with get_connection() as connection:
            connection.execute(
                """
                INSERT INTO user_shop_items (user_id, item_id, purchased_at)
                VALUES (?, ?, ?)
                """,
                (active_user["id"], item_id, timestamp),
            )
            connection.execute(
                """
                UPDATE users
                SET total_points = ?, last_badge = ?, updated_at = ?
                WHERE id = ?
                """,
                (new_points, new_badge, timestamp, active_user["id"]),
            )
    except sqlite3.IntegrityError:
        st.session_state.shop_feedback = ("info", "You already own this item.")
        return

    st.session_state.total_points = new_points
    st.session_state.badge = new_badge
    rank_label, rank_icon = determine_rank(new_points)
    st.session_state.rank_label = rank_label
    st.session_state.rank_icon = rank_icon
    st.session_state.purchased_items.append(item_id)

    update_leaderboard(active_user["username"], new_points)
    st.session_state.shop_feedback = ("success", f"Purchased {item['name']} for {cost} eco points!")
    load_active_user_from_db()
    refresh_shop_items()


def hash_password(password: str) -> str:
    salt = b"optimism-trash"
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 310000).hex()


def verify_password(password: str, password_hash: str) -> bool:
    return hash_password(password) == password_hash


def create_user(username: str, email: str, password: str) -> Tuple[bool, Optional[str]]:
    try:
        with get_connection() as connection:
            now = datetime.utcnow().isoformat()
            connection.execute(
                """
                INSERT INTO users (username, email, password_hash, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (username, email, hash_password(password), now, now),
            )
        return True, None
    except sqlite3.IntegrityError as error:
        if "username" in str(error).lower():
            return False, "Username already exists"
        if "email" in str(error).lower():
            return False, "Email already exists"
        return False, "Unable to create account at this time"


def get_user_by_credentials(username_or_email: str, password: str) -> Optional[sqlite3.Row]:
    with get_connection() as connection:
        query = "SELECT * FROM users WHERE username = ? OR email = ?"
        row = connection.execute(query, (username_or_email, username_or_email)).fetchone()
        if row and verify_password(password, row["password_hash"]):
            return row
    return None


def get_user_by_id(user_id: int) -> Optional[sqlite3.Row]:
    with get_connection() as connection:
        return connection.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()


def update_user_points(user_id: int, points: int, badge: str) -> None:
    with get_connection() as connection:
        now = datetime.utcnow().isoformat()
        connection.execute(
            """
            UPDATE users
            SET total_points = ?, last_badge = ?, updated_at = ?
            WHERE id = ?
            """,
            (points, badge, now, user_id),
        )


def fetch_user_purchases(user_id: int) -> List[int]:
    with get_connection() as connection:
        rows = connection.execute(
            "SELECT item_id FROM user_shop_items WHERE user_id = ?",
            (user_id,),
        ).fetchall()
    return [row["item_id"] for row in rows]


def set_active_user(row: sqlite3.Row) -> None:
    st.session_state[SESSION_KEY] = {
        "id": row["id"],
        "username": row["username"],
        "email": row["email"],
        "created_at": row["created_at"],
        "avatar": DEFAULT_AVATAR,
    }
    st.session_state.total_points = row["total_points"]
    st.session_state.badge = normalize_badge(row["total_points"], row["last_badge"])
    rank_label, rank_icon = determine_rank(row["total_points"])
    st.session_state.rank_label = rank_label
    st.session_state.rank_icon = rank_icon
    st.session_state.purchased_items = fetch_user_purchases(row["id"])
    ensure_shop_seeded()
    refresh_shop_items()
    if not st.session_state.shop_items:
        refresh_shop_items()
    if "leaderboard" not in st.session_state or not st.session_state.leaderboard:
        st.session_state.leaderboard = [
            {"name": row["username"], "points": row["total_points"]},
            *LEADERBOARD_SEED,
        ]
    else:
        st.session_state.leaderboard = [
            entry
            for entry in st.session_state.leaderboard
            if entry["name"].lower() != "you"
        ]
    update_leaderboard(row["username"], row["total_points"])
    st.session_state.auth_error = None


def clear_active_user() -> None:
    st.session_state[SESSION_KEY] = None
    st.session_state.total_points = 0
    st.session_state.badge = get_badge(st.session_state.total_points)
    rank_label, rank_icon = determine_rank(st.session_state.total_points)
    st.session_state.rank_label = rank_label
    st.session_state.rank_icon = rank_icon
    st.session_state.leaderboard = [
        {"name": "You", "points": st.session_state.total_points},
        *LEADERBOARD_SEED,
    ]
    st.session_state.purchased_items = []
    st.session_state.shop_items = []
    st.session_state.shop_initialized = False
    st.session_state.active_tab = "scanner"
    st.session_state.auth_error = None
    st.session_state.show_auth_form = "login"


def ensure_shop_seeded() -> None:
    if st.session_state.shop_initialized:
        return

    with get_connection() as connection:
        existing_count = connection.execute("SELECT COUNT(*) FROM shop_items").fetchone()[0]
        if existing_count == 0:
            connection.executemany(
                """
                INSERT INTO shop_items (name, item_type, cost, description, icon, is_active)
                VALUES (?, ?, ?, ?, ?, 1)
                """,
                [
                    (
                        item["name"],
                        item["item_type"],
                        item["cost"],
                        item["description"],
                        item.get("icon"),
                    )
                    for item in SHOP_ITEMS_SEED
                ],
            )
        st.session_state.shop_initialized = True

    refresh_shop_items()


def load_active_user_from_db() -> None:
    active_user = st.session_state.get(SESSION_KEY)
    if not active_user:
        return
    row = get_user_by_id(active_user["id"])
    if row is None:
        clear_active_user()
        return
    set_active_user(row)


def render_profile_card(user: Optional[Dict[str, Any]]) -> None:
    container = st.container()
    if user:
        _, sign_out_column = container.columns([0.75, 0.25])
        with sign_out_column:
            if st.button("Sign out", key="sign_out_button"):
                clear_active_user()
                st.rerun()

   
    if user:
        username = user.get("username", "")
        initial = username[:1].upper() if username else "?"
        total_points = st.session_state.get("total_points", 0)
        rank_label = st.session_state.get("rank_label")
        rank_icon = st.session_state.get("rank_icon")
        if not rank_label or not rank_icon:
            rank_label, rank_icon = determine_rank(total_points)
        gradient = "linear-gradient(135deg, #34d399, #0ea5e9)"
        points_display = f"{total_points:,} pts"
        container.markdown(
            f"""
            <div style="text-align:right;">
                <div style="width:64px;height:64px;border-radius:50%;border:2px solid rgba(12, 91, 61, 0.2);margin-bottom:0.4rem;background:{gradient};box-shadow:0 2px 8px rgba(0, 0, 0, 0.12);display:inline-flex;align-items:center;justify-content:center;color:#ffffff;font-weight:700;font-size:1.75rem;">
                    {initial}
                </div>
                <p style="margin:0;"><strong>{username}</strong></p>
                <p style="margin:0;">{user['email']}</p>
                <p style="margin:0.35rem 0 0.6rem;font-size:0.95rem;display:flex;align-items:center;gap:0.4rem;justify-content:flex-end;">
                    <span style="font-size:1.15rem;">{rank_icon}</span>
                    <strong>{points_display}</strong>
                    <span style="color:rgba(11, 93, 58, 0.85);font-weight:600;">{rank_label}</span>
                </p>
                <p style="margin:0 0 0.8rem;">Password: {MASKED_PASSWORD_DISPLAY}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        print("")


def render_login_form() -> None:
    with st.form("login_form"):
        username_or_email = st.text_input("Username or Email", key="login_username_or_email").strip()
        password = st.text_input("Password", type="password", key="login_password")
        submitted = st.form_submit_button("Sign in")
        if submitted:
            if not username_or_email or not password:
                st.session_state.auth_error = "Enter both your username/email and password."
                return
            user_row = get_user_by_credentials(username_or_email, password)
            if user_row is None:
                st.session_state.auth_error = "Invalid credentials. Please try again."
                return
            set_active_user(user_row)
            st.success("Signed in successfully.")
            st.rerun()


def render_signup_form() -> None:
    with st.form("signup_form"):
        username = st.text_input("Username", key="signup_username").strip()
        email = st.text_input("Email", key="signup_email").strip()
        password = st.text_input("Password", type="password", key="signup_password")
        confirm_password = st.text_input("Confirm Password", type="password", key="signup_confirm_password")
        submitted = st.form_submit_button("Create account")
        if submitted:
            if not username or not email or not password or not confirm_password:
                st.session_state.auth_error = "All fields are required to create an account."
                return
            if "@" not in email or "." not in email.split("@")[-1]:
                st.session_state.auth_error = "Please enter a valid email address."
                return
            if password != confirm_password:
                st.session_state.auth_error = "Passwords do not match."
                return
            if len(password) < 6:
                st.session_state.auth_error = "Choose a password with at least 6 characters."
                return
            success, message = create_user(username, email, password)
            if not success:
                st.session_state.auth_error = message
                return
            st.session_state.auth_error = None
            st.session_state.show_auth_form = "login"
            st.session_state.login_username_or_email = username
            for key in (
                "signup_username",
                "signup_email",
                "signup_password",
                "signup_confirm_password",
            ):
                st.session_state[key] = ""
            st.success("Account created! Sign in to start using the app.")


def render_auth_forms() -> None:
    st.markdown("### Access your eco account")
    option_labels = ["Sign in", "Create account"]
    current_mode = st.session_state.get("show_auth_form", "login")
    default_index = 0 if current_mode == "login" else 1
    selected_label = st.radio(
        "Choose access mode",
        options=option_labels,
        index=default_index,
        horizontal=True,
        key="auth_mode_selector",
        label_visibility="collapsed",
    )
    selected_mode = "login" if selected_label == "Sign in" else "signup"
    if selected_mode != current_mode:
        st.session_state.auth_error = None
    st.session_state.show_auth_form = selected_mode

    if st.session_state.show_auth_form == "login":
        render_login_form()
    else:
        render_signup_form()

    if st.session_state.auth_error:
        st.error(st.session_state.auth_error)


def ensure_shop_ready() -> None:
    ensure_shop_seeded()
    if not st.session_state.shop_items:
        refresh_shop_items()
    if not st.session_state.shop_item_lookup:
        st.session_state.shop_item_lookup = {
            item["id"]: item for item in st.session_state.shop_items
        }




def render_scanner_and_analysis() -> None:
    left_column, right_column = st.columns([1.5, 1])

    with left_column:
        st.markdown('<div class="eco-card" style="background:none;border:none;box-shadow:none;padding:0;">', unsafe_allow_html=True)
        st.markdown('<h3 class="step-heading"> Capture or Upload</h3>', unsafe_allow_html=True)
        camera_input = st.camera_input(
            "Use your camera to capture an item",
            help="On desktop browsers this may prompt a file selection dialog if the camera is blocked.",
        )

        image_bytes = None
        image_source = None
        if camera_input is not None:
            image_bytes = camera_input.getvalue()
            image_source = "camera"

        analyze_triggered = st.button("Analyze Image", type="primary")

        if analyze_triggered and image_bytes is None:
            st.warning("Please capture or upload an image before analyzing.")

        if analyze_triggered and image_bytes is not None:
            with st.spinner("Running object detection and classification…"):
                try:
                    _ = get_detector()
                    detections = detect_objects(image_bytes)
                    primary_model = get_mobilenet()
                    result = analyze_image(primary_model, None, image_bytes)
                    st.session_state.last_result = {
                        **result,
                        "emissions": asdict(result["emissions"]),
                        "source": image_source,
                        "detections": [detection.as_dict() for detection in detections],
                    }
                    points_awarded = st.session_state.last_result["emissions"]["points_awarded"]
                    st.session_state.total_points = max(
                        st.session_state.total_points + points_awarded,
                        0,
                    )
                    st.session_state.badge = get_badge(st.session_state.total_points)
                    rank_label, rank_icon = determine_rank(st.session_state.total_points)
                    st.session_state.rank_label = rank_label
                    st.session_state.rank_icon = rank_icon

                    active_user = st.session_state.get(SESSION_KEY)
                    leaderboard_name = "You"
                    if active_user:
                        leaderboard_name = active_user["username"]
                        update_user_points(
                            user_id=active_user["id"],
                            points=st.session_state.total_points,
                            badge=st.session_state.badge,
                        )
                        load_active_user_from_db()

                    update_leaderboard(leaderboard_name, st.session_state.total_points)
                    st.success("Analysis complete! Scroll down for insights and leaderboard updates.")
                except MissingDependencyError as error:
                    st.error(str(error))
                    st.session_state.last_result = None
                except ClassificationError as error:
                    st.error(f"Classification failed: {error}")
                except Exception as error:  # noqa: BLE001
                    st.exception(error)

        st.markdown('</div>', unsafe_allow_html=True)

    with right_column:
        st.markdown('<div class="eco-card" style="background:none;border:none;box-shadow:none;padding:0;">', unsafe_allow_html=True)
        st.markdown('<h3 class="step-heading"> Your Eco Impact</h3>', unsafe_allow_html=True)

        metric_col, badge_col = st.columns(2)
        metric_col.metric("Total Eco Points", st.session_state.total_points)
        badge_col.metric("Badge", st.session_state.badge)

        last_result = st.session_state.last_result
        if last_result is None:
            st.info("Capture or upload an image to see nature-positive insights here.")
        else:
            emissions_info = last_result["emissions"]

            st.markdown(
                f"<h4 style='color:#0b5d3a;'>Identified Item · {last_result['display_name']}</h4>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<p><strong>Category:</strong> <code>{last_result['category']}</code></p>",
                unsafe_allow_html=True,
            )

            st.markdown(
                f"<p><strong>Estimated CO₂e:</strong> {emissions_info['kg_co2e']:.2f} kg · <em>{'live data' if emissions_info['source'] == 'climatiq' else emissions_info['source']}</em></p>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<p><strong>Eco Points Earned:</strong> {emissions_info['points_awarded']}</p>",
                unsafe_allow_html=True,
            )

            if emissions_info["points_awarded"] > 0:
                st.success("Nice work! Lower emissions detected — energy savings unlocked.")
            else:
                st.warning("Higher emissions detected. Consider reusable or low-energy alternatives next time.")

            detections: List[Dict[str, float]] = last_result.get("detections", [])
            if detections:
                st.markdown(
                    """
                    <div style="border:1px solid #0b5d3a1a;border-radius:12px;padding:0.85rem 1.1rem;margin-top:1.1rem;background:rgba(52, 211, 153, 0.07);">
                        <p style="margin:0 0 0.65rem;font-weight:600;color:#0b5d3a;">Objects Detected</p>
                        <p style="margin:0;color:rgba(11, 67, 49, 0.75);">Detection details are hidden to keep the interface tidy.</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        """
        <div class="tips-card">
            <h3>Energy-Savvy Tips</h3>
            <ul>
                <li><strong>Capture in daylight</strong> to reduce the need for flash and save device energy.</li>
                <li><strong>Reuse and refill</strong>: choose durable containers over single-use plastic.</li>
                <li><strong>Compost organics</strong> to turn scraps into soil rather than methane.</li>
                <li><strong>Power down</strong> devices after scanning to cut phantom energy waste.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_shop(active_user: Dict[str, Any]) -> None:
    """Display shop inventory, ownership status, and purchase controls."""
    
    st.caption("Spend your eco points on profile flair and ambient effects.")

    username = active_user.get("username", "Eco hero")
    st.markdown(f"Welcome back, **{username}**! Redeem upgrades to celebrate your sustainability wins.")

    summary_col1, summary_col2 = st.columns(2)
    with summary_col1:
        st.metric("Eco Points Available", st.session_state.total_points)
    with summary_col2:
        st.metric("Items Owned", len(st.session_state.purchased_items))

    feedback = st.session_state.get("shop_feedback")
    if feedback:
        feedback_type, feedback_message = feedback
        feedback_renderer = getattr(st, feedback_type, st.info)
        feedback_renderer(feedback_message)
        st.session_state.shop_feedback = None

    items = st.session_state.shop_items
    if not items:
        st.info("The shop is restocking new items. Check back soon!")
        return

    item_columns = st.columns(2)
    purchased_items = set(st.session_state.purchased_items)

    for index, item in enumerate(items):
        column = item_columns[index % len(item_columns)]
        with column:
            owned = item["id"] in purchased_items
            cost = item["cost"]
            affordable = st.session_state.total_points >= cost

            st.markdown(f"### {item.get('icon') or '🛍️'} {item['name']}")
            st.caption(f"{item['item_type'].capitalize()} · {cost} eco points")
            st.write(item["description"])

            if owned:
                st.success("Already owned and ready to equip.")
            elif not affordable:
                deficit = cost - st.session_state.total_points
                st.warning(f"Earn {deficit} more eco points to unlock this item.")
            else:
                st.caption("You have enough eco points to purchase this item.")

            button_label = "Owned" if owned else f"Purchase for {cost} pts"
            if st.button(
                button_label,
                key=f"purchase_{item['id']}",
                disabled=owned,
                use_container_width=True,
            ):
                purchase_item(item["id"])
                st.rerun()

            st.divider()


def calculate_badge(total_points: int) -> str:
    label, icon = determine_rank(total_points)
    return f"{icon} {label}"


@st.cache_resource(show_spinner=True)
def get_mobilenet():
    """Load MobileNetV2 once and reuse across reruns."""
    return load_classifier()


@st.cache_resource(show_spinner=True)
def get_blip():
    """Load BLIP once the checkpoint is available."""
    return load_blip()


@st.cache_resource(show_spinner=True)
def get_detector():
    """Load YOLO once for consistent detections."""
    return load_yolo()


def get_badge(total_points: int) -> str:
    """Map points to a badge label."""
    return calculate_badge(total_points)


def update_leaderboard(name: str, points: int) -> None:
    """Insert or update a leaderboard entry for the user."""
    board = list(st.session_state.leaderboard)
    for entry in board:
        if entry["name"].lower() == name.lower():
            entry["points"] = points
            break
    else:
        board.append({"name": name, "points": points})
    board.sort(key=lambda item: item["points"], reverse=True)
    st.session_state.leaderboard = board[:5]


initialize_state()

# Ensure the database file exists early so first-time runs
# do not hit race conditions when opening connections.
DATABASE_FILE.touch(exist_ok=True)

st.markdown(
    """
    <div class="main-title">
        <span class="leaf-icon">🌿</span>
        <span>GreenScan</span>
    </div>
    """,
    unsafe_allow_html=True,
)

header_left, header_right = st.columns([1.8, 1])
with header_left:
    st.title("Optimism for a Greener World")
    st.subheader("Capture waste, reclaim energy savings, and grow your eco impact.")
    st.write(
        "Snap or upload a photo of waste, uncover its carbon footprint, and unlock greener habits, along with a a better rank. "
       
    )

with header_right:
    render_profile_card(st.session_state.get(SESSION_KEY))

active_user = st.session_state.get(SESSION_KEY)
if active_user:
    ensure_shop_ready()
else:
    st.session_state.active_tab = "scanner"
if active_user is None:
    auth_expander = st.expander("Sign in or create account", expanded=True)
    with auth_expander:
        render_auth_forms()
    st.info("Sign in or create an account to access the Optimism Trash Classifier features.")
    st.stop()

nav_choices = [
    ("scanner", "♻️ Scanner"),
    
]
choice_to_label = {value: label for value, label in nav_choices}
label_to_choice = {label: value for value, label in nav_choices}
available_choices = [value for value, _ in nav_choices]

current_tab = st.session_state.get("active_tab", "scanner")
if current_tab not in available_choices:
    current_tab = "scanner"
    st.session_state.active_tab = current_tab

selected_label = st.radio(
    "Choose what to explore",
    options=[label for _, label in nav_choices],
    index=available_choices.index(current_tab),
    horizontal=True,
    key="main_tab_selector",
)

selected_tab = label_to_choice[selected_label]
st.session_state.active_tab = selected_tab

if selected_tab == "scanner":
    render_scanner_and_analysis()
else:
    render_shop(active_user)