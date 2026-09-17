#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import io
import json
import logging
import os
import re
import sys
import threading
import sqlite3
import uuid
import math
from datetime import datetime, date, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np
import pandas as pd
from tabulate import tabulate

from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

import uvicorn

# =============================================================================
# SETUP
# =============================================================================

sys.setrecursionlimit(10000)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("llm_analyser")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

os.makedirs("templates", exist_ok=True)
os.makedirs("static", exist_ok=True)
os.makedirs("static/css", exist_ok=True)
os.makedirs("static/js", exist_ok=True)

# =============================================================================
# NEB Grading System
# =============================================================================

GRADING_SYSTEM = [
    {"min_percentage": 90, "max_percentage": 100, "grade": "A+", "gpa": 4.00},
    {"min_percentage": 80, "max_percentage": 89.99, "grade": "A", "gpa": 3.60},
    {"min_percentage": 70, "max_percentage": 79.99, "grade": "B+", "gpa": 3.20},
    {"min_percentage": 60, "max_percentage": 69.99, "grade": "B", "gpa": 2.80},
    {"min_percentage": 50, "max_percentage": 59.99, "grade": "C+", "gpa": 2.40},
    {"min_percentage": 40, "max_percentage": 49.99, "grade": "C", "gpa": 2.00},
    {"min_percentage": 35, "max_percentage": 39.99, "grade": "D", "gpa": 1.60},
    {"min_percentage": 0, "max_percentage": 34.99, "grade": "NG", "gpa": 0.00},
]

GRADE_TO_GPA_MAP = {
    'A+': 4.00, 'A': 3.60, 'B+': 3.20, 'B': 2.80,
    'C+': 2.40, 'C': 2.00, 'D': 1.60, 'NG': 0.00
}


def get_grade_from_percentage(percentage: float) -> str:
    if percentage is None or (isinstance(percentage, float) and math.isnan(percentage)):
        return "NG"
    percentage = min(max(percentage, 0), 100)
    for rule in GRADING_SYSTEM:
        if rule["min_percentage"] <= percentage <= rule["max_percentage"]:
            return rule["grade"]
    return "NG"


def get_gpa_from_percentage(percentage: float) -> float:
    if percentage is None or (isinstance(percentage, float) and math.isnan(percentage)):
        return 0.0
    percentage = min(max(percentage, 0), 100)
    for rule in GRADING_SYSTEM:
        if rule["min_percentage"] <= percentage <= rule["max_percentage"]:
            return rule["gpa"]
    return 0.0


def get_grade_band(percentage: float) -> Dict[str, Any]:
    """Return the full grading-system row (min/max/grade/gpa) a percentage falls into."""
    percentage = min(max(percentage or 0, 0), 100)
    for rule in GRADING_SYSTEM:
        if rule["min_percentage"] <= percentage <= rule["max_percentage"]:
            return rule
    return GRADING_SYSTEM[-1]


def get_next_grade_band(current_band: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """GRADING_SYSTEM is ordered highest-to-lowest, so 'next up' is the previous index."""
    idx = GRADING_SYSTEM.index(current_band)
    if idx == 0:
        return None
    return GRADING_SYSTEM[idx - 1]


def calculate_age_from_dob(dob_str: str) -> int:
    if not dob_str or dob_str == "N/A":
        return 0
    try:
        parts = dob_str.split('-')
        if len(parts) == 3:
            year = int(parts[0])
            current_year = 2082
            return current_year - year
    except Exception:
        pass
    return 0


def convert_to_serializable(obj: Any) -> Any:
    if isinstance(obj, pd.DataFrame):
        return json.loads(obj.to_json(orient="records", date_format="iso", default_handler=str))
    if isinstance(obj, pd.Series):
        return convert_to_serializable(obj.to_dict())
    if isinstance(obj, (pd.Timestamp, datetime, date)):
        return obj.isoformat()
    if isinstance(obj, Decimal):
        val = float(obj)
        if math.isnan(val) or math.isinf(val):
            return None
        return val
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return float(obj)
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {str(k): convert_to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [convert_to_serializable(v) for v in obj]
    if pd.isna(obj):
        return None
    return obj


# =============================================================================
# SQL Functions for Grading
# =============================================================================

def sql_get_grade(percentage: Any) -> Optional[str]:
    if percentage is None:
        return None
    try:
        pct = float(percentage)
        return get_grade_from_percentage(pct)
    except (TypeError, ValueError):
        return None


def sql_get_gpa(percentage: Any) -> Optional[float]:
    if percentage is None:
        return None
    try:
        pct = float(percentage)
        return get_gpa_from_percentage(pct)
    except (TypeError, ValueError):
        return None


def sql_calculate_age(dob_str: Any) -> Optional[int]:
    if dob_str is None:
        return None
    try:
        return calculate_age_from_dob(str(dob_str))
    except Exception:
        return None


def sql_get_percentage(obtained: Any, full: Any) -> Optional[float]:
    if obtained is None or full is None:
        return None
    try:
        o = float(obtained)
        f = float(full)
        if f == 0:
            return None
        return (o / f) * 100
    except Exception:
        return None


def sql_is_passed(obtained: Any, pass_mark: Any) -> Optional[int]:
    if obtained is None or pass_mark is None:
        return None
    try:
        o = float(obtained)
        p = float(pass_mark)
        return 1 if o >= p else 0
    except Exception:
        return None


# =============================================================================
# FASTAPI APP
# =============================================================================

app = FastAPI(
    title="Complete Education Data Analysis System",
    description="Analyze student data with full semantic query handling",
    version="9.0-dynamic"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# =============================================================================
# SESSION MANAGER
# =============================================================================

class SessionManager:
    def __init__(self):
        self.sessions: Dict[str, Dict] = {}
        self._lock = threading.Lock()
        self._cleanup_thread = None
        self._running = True
        self._start_cleanup_thread()

    def _start_cleanup_thread(self):
        def cleanup_worker():
            while self._running:
                try:
                    self._cleanup_expired_sessions()
                except Exception as e:
                    logger.error(f"Session cleanup error: {e}")
                import time
                time.sleep(300)
        self._cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        self._cleanup_thread.start()

    def _cleanup_expired_sessions(self):
        with self._lock:
            current_time = datetime.now()
            expired = []
            for session_id, session in self.sessions.items():
                last_activity = session.get("last_activity", session.get("created_at"))
                if current_time - last_activity > timedelta(hours=1):
                    expired.append(session_id)
            for session_id in expired:
                del self.sessions[session_id]
                logger.info(f"Cleaned up expired session: {session_id}")

    def create_session(self) -> str:
        session_id = str(uuid.uuid4())
        with self._lock:
            self.sessions[session_id] = {
                "created_at": datetime.now(),
                "last_activity": datetime.now(),
                "datasets": {},
                "active_dataset": None,
                "history": []
            }
        logger.info(f"Created new session: {session_id}")
        return session_id

    def get_session(self, session_id: str) -> Optional[Dict]:
        with self._lock:
            session = self.sessions.get(session_id)
            if session:
                session["last_activity"] = datetime.now()
            return session

    def update_session_activity(self, session_id: str):
        with self._lock:
            if session_id in self.sessions:
                self.sessions[session_id]["last_activity"] = datetime.now()

    def set_dataset(self, session_id: str, dataset_name: str, tables: Dict, rel_map: Any):
        with self._lock:
            if session_id in self.sessions:
                self.sessions[session_id]["datasets"][dataset_name] = {
                    "tables": tables,
                    "rel_map": rel_map,
                    "uploaded_at": datetime.now()
                }
                self.sessions[session_id]["active_dataset"] = dataset_name
                self.sessions[session_id]["last_activity"] = datetime.now()

    def get_dataset(self, session_id: str, dataset_name: str) -> Optional[Tuple[Dict, Any]]:
        with self._lock:
            session = self.sessions.get(session_id)
            if session and dataset_name in session["datasets"]:
                session["last_activity"] = datetime.now()
                data = session["datasets"][dataset_name]
                return data["tables"], data["rel_map"]
        return None, None

    def list_datasets(self, session_id: str) -> List[str]:
        with self._lock:
            session = self.sessions.get(session_id)
            if session:
                session["last_activity"] = datetime.now()
                return list(session["datasets"].keys())
        return []

    def add_to_history(self, session_id: str, query: str, response: Dict):
        with self._lock:
            if session_id in self.sessions:
                self.sessions[session_id]["history"].append({
                    "query": query,
                    "response": response,
                    "timestamp": datetime.now().isoformat()
                })
                if len(self.sessions[session_id]["history"]) > 100:
                    self.sessions[session_id]["history"] = self.sessions[session_id]["history"][-100:]

    def get_history(self, session_id: str) -> List[Dict]:
        with self._lock:
            session = self.sessions.get(session_id)
            if session:
                return session.get("history", [])
            return []

    def delete_session(self, session_id: str):
        with self._lock:
            if session_id in self.sessions:
                del self.sessions[session_id]
                logger.info(f"Deleted session: {session_id}")

    def get_session_count(self) -> int:
        with self._lock:
            return len(self.sessions)

    def shutdown(self):
        self._running = False
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=5)


session_manager = SessionManager()

# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class QueryRequest(BaseModel):
    query: str
    dataset_name: str


class SaveChatPayload(BaseModel):
    json: Dict[str, Any]
    fileName: str


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_session_id(request: Request) -> Optional[str]:
    session_id = request.cookies.get("session_id")
    if session_id and session_manager.get_session(session_id):
        return session_id
    return None


def create_session_response(response: Response, session_id: str):
    response.set_cookie(
        key="session_id",
        value=session_id,
        httponly=True,
        max_age=3600 * 24 * 7,
        samesite="lax",
        path="/"
    )


# =============================================================================
# MIDDLEWARE
# =============================================================================

@app.middleware("http")
async def session_middleware(request: Request, call_next):
    session_id = request.cookies.get("session_id")
    if session_id:
        session_manager.update_session_activity(session_id)
    response = await call_next(request)
    return response


# =============================================================================
# RELATIONSHIP MAP
# =============================================================================

@dataclass
class Relationship:
    source_table: str
    source_field: str
    target_table: str
    target_field: str
    cardinality: str
    description: str = ""
    confidence: float = 1.0

    def join_condition(self) -> str:
        return f"{self.source_table}.{self.source_field} = {self.target_table}.{self.target_field}"


@dataclass
class RelationshipMap:
    relationships: List[Relationship] = field(default_factory=list)
    table_counts: Dict[str, int] = field(default_factory=dict)
    table_schemas: Dict[str, Dict] = field(default_factory=dict)
    table_samples: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tables": list(self.table_counts.keys()),
            "table_counts": self.table_counts,
            "relationships": [
                {
                    "source_table": r.source_table,
                    "source_field": r.source_field,
                    "target_table": r.target_table,
                    "target_field": r.target_field,
                    "cardinality": r.cardinality,
                    "description": r.description
                }
                for r in self.relationships
            ],
            "schemas": self.table_schemas
        }


# =============================================================================
# DYNAMIC PARSER
# =============================================================================

def make_hashable(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return tuple(make_hashable(v) for v in value)
    if isinstance(value, dict):
        return tuple((k, make_hashable(v)) for k, v in sorted(value.items()))
    return value


def normalize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ['uuid', 'student_id', 'student_uuid']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    for col in df.columns:
        if col.endswith('_id') or col == 'id':
            converted = pd.to_numeric(df[col], errors='coerce')
            if converted.notna().sum() >= df[col].notna().sum() * 0.8:
                df[col] = converted
    df = df.replace({np.nan: None, pd.NA: None})
    return df


class DynamicParser:
    def __init__(self):
        self.relationship_map = RelationshipMap()

    def parse(self, data: Any) -> Dict[str, pd.DataFrame]:
        result: Dict[str, pd.DataFrame] = {}
        self._extract_all_tables(data, result)
        self._build_schemas(result)
        self._infer_relationships(result)
        return result

    def _extract_all_tables(self, data: Any, result: Dict[str, pd.DataFrame], prefix: str = ""):
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, list) and len(value) > 0:
                    if isinstance(value[0], dict):
                        table_name = self._safe_name(key)
                        df = self._list_to_df(value)
                        if df is not None and len(df) > 0:
                            result[table_name] = normalize_dtypes(df)
                            self.relationship_map.table_counts[table_name] = len(df)
                elif isinstance(value, dict):
                    self._extract_all_tables(value, result, key)
        elif isinstance(data, list) and len(data) > 0:
            if isinstance(data[0], dict):
                table_name = prefix if prefix else "data"
                df = self._list_to_df(data)
                if df is not None and len(df) > 0:
                    result[table_name] = normalize_dtypes(df)
                    self.relationship_map.table_counts[table_name] = len(df)

    def _list_to_df(self, items: List[Any]) -> Optional[pd.DataFrame]:
        if not items or not isinstance(items[0], dict):
            return None
        rows = []
        for idx, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            row = {}
            for k, v in item.items():
                if isinstance(v, (dict, list)):
                    continue
                try:
                    make_hashable(v)
                    row[k] = v
                except Exception:
                    row[k] = str(v)
            row["_index"] = idx
            rows.append(row)
        return pd.DataFrame(rows) if rows else None

    @staticmethod
    def _safe_name(key: str) -> str:
        name = re.sub(r'[^a-zA-Z0-9_]', '_', str(key)).strip('_') or "table"
        name = re.sub(r'(?<!^)(?<![A-Z_])([A-Z])', r'_\1', name)
        return name.lower()

    def _build_schemas(self, tables: Dict[str, pd.DataFrame]):
        for table_name, df in tables.items():
            schema = {}
            for col in df.columns:
                if col == "_index":
                    continue
                try:
                    unique_vals = int(df[col].nunique())
                except Exception:
                    unique_vals = len(df[col].dropna())
                schema[col] = {
                    "type": str(df[col].dtype),
                    "non_null": int(df[col].count()),
                    "unique": unique_vals,
                    "sample": df[col].head(3).tolist() if len(df) > 0 else [],
                }
            self.relationship_map.table_schemas[table_name] = schema
            if len(df) > 0:
                self.relationship_map.table_samples[table_name] = df.head(3).to_string()

    def _infer_relationships(self, tables: Dict[str, pd.DataFrame]):
        table_names = list(tables.keys())
        seen = set()

        known_chain = [
            ("students", "uuid", "marks", "student_id", "One-to-Many", "Students have marks"),
            ("marks", "exam_parameter_id", "exam_parameters", "id", "Many-to-One", "Marks reference exam parameters"),
            ("exam_parameters", "exam_id", "exams", "id", "Many-to-One", "Exam parameters reference exams"),
            ("exam_parameters", "class_section_subject_id", "subjects", "cssId", "Many-to-One", "Exam parameters reference subjects"),
            ("students", "id", "attendances", "student_id", "One-to-Many", "Students have attendance"),
            ("students", "id", "symbol_no", "student_id", "One-to-One", "Students have symbol numbers"),
            ("students", "section_id", "sections", "id", "Many-to-One", "Students belong to sections"),
        ]
        for src_t, src_f, tgt_t, tgt_f, card, desc in known_chain:
            if src_t in tables and tgt_t in tables and src_f in tables[src_t].columns and tgt_f in tables[tgt_t].columns:
                key = (src_t, src_f, tgt_t, tgt_f)
                if key not in seen:
                    seen.add(key)
                    self.relationship_map.relationships.append(
                        Relationship(src_t, src_f, tgt_t, tgt_f, card, desc, confidence=1.0))


def read_dataset_file(file_path: str) -> Tuple[Dict[str, pd.DataFrame], RelationshipMap]:
    ext = file_path.lower().rsplit(".", 1)[-1] if "." in file_path else ""
    if ext == "json":
        with open(file_path, 'rb') as f:
            content = f.read()
        data = json.loads(content.decode('utf-8'))
        parser = DynamicParser()
        tables = parser.parse(data)
        return tables, parser.relationship_map
    if ext == "csv":
        content = open(file_path, 'rb').read()
        for enc in ["utf-8", "utf-8-sig", "latin1", "cp1252"]:
            try:
                df = pd.read_csv(io.BytesIO(content), encoding=enc)
                return {os.path.basename(file_path).rsplit('.', 1)[0]: df}, RelationshipMap()
            except Exception:
                continue
        raise ValueError("Unable to parse CSV")
    if ext in ["xlsx", "xls"]:
        content = open(file_path, 'rb').read()
        xls = pd.ExcelFile(io.BytesIO(content))
        tables = {sheet: pd.read_excel(xls, sheet_name=sheet) for sheet in xls.sheet_names}
        return tables, RelationshipMap()
    raise ValueError(f"Unsupported file format: .{ext}")


# =============================================================================
# SQL GENERATOR  (validate_sql restored + enforced)
# =============================================================================

class SQLGenerator:
    def __init__(self, rel_map: RelationshipMap):
        self.rel_map = rel_map
        self.connection = sqlite3.connect(':memory:')
        self.connection.execute('PRAGMA case_sensitive_like = OFF')
        self.connection.create_function("GET_GRADE", 1, sql_get_grade)
        self.connection.create_function("GET_GPA", 1, sql_get_gpa)
        self.connection.create_function("CALCULATE_AGE", 1, sql_calculate_age)
        self.connection.create_function("GET_PERCENTAGE", 2, sql_get_percentage)
        self.connection.create_function("IS_PASSED", 2, sql_is_passed)
        self.last_sql = None
        self.last_error = None

    def create_tables_from_dataframes(self, dataframes: Dict[str, pd.DataFrame]):
        for table_name, df in dataframes.items():
            df.to_sql(table_name, self.connection, if_exists='replace', index=False)
        self.connection.commit()

    @staticmethod
    def validate_sql(sql: str) -> Tuple[bool, str]:
        sql_clean = sql.strip().rstrip(';')
        if not sql_clean:
            return False, "Empty SQL query"
        sql_upper = sql_clean.upper()
        for op in ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'CREATE',
                   'TRUNCATE', 'ATTACH', 'PRAGMA', 'REPLACE', 'VACUUM']:
            if re.search(rf'\b{op}\b', sql_upper):
                return False, f"Operation '{op}' is not allowed"
        if not re.match(r'^\s*(SELECT|WITH)\b', sql_clean, re.IGNORECASE):
            return False, "Query must start with SELECT or WITH"
        if sql_clean.count('(') != sql_clean.count(')'):
            return False, "Unbalanced parentheses"
        if 'SELECT' in sql_upper and 'FROM' not in sql_upper:
            return False, "SELECT query missing FROM clause"
        return True, "Valid SQL"

    def execute_query(self, sql: str, validate: bool = False) -> pd.DataFrame:
        if validate:
            ok, msg = self.validate_sql(sql)
            if not ok:
                self.last_error = msg
                raise ValueError(f"SQL rejected: {msg}")
        try:
            self.last_sql = sql
            self.last_error = None
            return pd.read_sql_query(sql, self.connection)
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"SQL execution error: {e}")
            raise


# =============================================================================
# SUBJECT FETCHER
# =============================================================================

class SubjectFetcher:
    @staticmethod
    def get_all_subjects(sql_generator: SQLGenerator) -> List[str]:
        try:
            query = "SELECT title FROM subjects ORDER BY title"
            df = sql_generator.execute_query(query)
            return df['title'].tolist() if not df.empty else []
        except Exception as e:
            logger.warning(f"Could not fetch subjects: {e}")
            return []

    @staticmethod
    def get_subject_condition(sql_generator: SQLGenerator, subject_query: str) -> str:
        subjects = SubjectFetcher.get_all_subjects(sql_generator)
        if not subjects:
            return f"UPPER(sub.title) LIKE UPPER('%{subject_query}%')"

        subject_lower = subject_query.lower().strip()
        matching_subjects = []

        for subject in subjects:
            if subject_lower in subject.lower() or subject.lower() in subject_lower:
                matching_subjects.append(subject)
            else:
                subject_words = set(subject.lower().split())
                query_words = set(subject_lower.split())
                if len(query_words & subject_words) >= 1:
                    matching_subjects.append(subject)

        if not matching_subjects:
            return f"UPPER(sub.title) LIKE UPPER('%{subject_query}%')"

        conditions = [f"UPPER(sub.title) LIKE UPPER('%{s}%')" for s in matching_subjects]
        return "(" + " OR ".join(conditions) + ")"


# =============================================================================
# STUDENT RESOLVER  (NEW  resolves a name to a uuid BEFORE any join)
# =============================================================================

class StudentResolver:
    @staticmethod
    def resolve_uuid(sql_gen: 'SQLGenerator', name: str) -> Optional[Tuple[str, str]]:
        """Returns (uuid, matched_display_name) or None if no student matches."""
        clean = (name or "").strip()
        if not clean:
            return None
        sql = f"""
        SELECT uuid, name FROM students
        WHERE LOWER(TRIM(name)) LIKE LOWER(TRIM('%{clean}%'))
        LIMIT 2
        """
        try:
            df = sql_gen.execute_query(sql)
        except Exception as e:
            logger.error(f"Name resolution error for '{name}': {e}")
            return None
        if df.empty:
            return None
        if len(df) > 1:
            logger.warning(
                f"Ambiguous name '{name}' matched {len(df)} students; using first: {df.iloc[0]['name']}"
            )
        return df.iloc[0]['uuid'], df.iloc[0]['name']


# =============================================================================
# UNIVERSAL SEMANTIC QUERY PARSER
# =============================================================================

class UniversalSemanticQueryParser:
    """Complete semantic query parser that handles ALL query types"""

    SEMANTIC_MAPPINGS = {
        'help': {
            'keywords': ['hello', 'hi', 'hey', 'greetings', 'good morning', 'good afternoon', 'good evening',
                         'help', 'assist', 'support', 'guide', 'how to', 'what can', 'capabilities', 'features',
                         'welcome', 'namaste', 'what can you do'],
            'patterns': [
                r'^(?:hello|hi|hey|greetings|good\s+(?:morning|afternoon|evening)|namaste)',
                r'(?:help|assist|support|guide)\s+me',
                r'how\s+to\s+(?:use|query|ask)',
                r'what\s+can\s+you\s+do',
                r'capabilities',
                r'features',
            ]
        },
        'student_details': {
            'keywords': ['details', 'info', 'information', 'profile', 'who is', 'tell me about', 'show me',
                         'find', 'get', 'about', 'describe', 'student'],
            'patterns': [
                r'(?:details|info|information|profile)\s+(?:of|about|for|on)\s+([a-z][a-z\s]+)',
                r'(?:who|what)\s+is\s+([a-z][a-z\s]+)',
                r'tell\s+me\s+about\s+([a-z][a-z\s]+)',
                r'show\s+me\s+([a-z][a-z\s]+)',
                r'find\s+([a-z][a-z\s]+)',
                r'profile\s+of\s+([a-z][a-z\s]+)',
                r'about\s+([a-z][a-z\s]+)',
            ]
        },
        'student_list': {
            'keywords': ['list', 'all', 'every', 'show', 'display', 'students', 'class', 'directory', 'roster'],
            'patterns': [
                r'(?:list|show|display|get)\s+(?:all|every)\s+(?:students|student|class)',
                r'(?:students|student)\s+(?:list|roster|directory)',
                r'who\s+are\s+(?:all|the)\s+students',
                r'show\s+me\s+(?:all|the)\s+students',
            ]
        },
        'student_count': {
            'keywords': ['count', 'number', 'total', 'how many', 'strength', 'population', 'size', 'enrollment'],
            'patterns': [
                r'(?:how many|count|number of|total)\s+(?:students|student)',
                r'(?:class|section)\s+(?:strength|size|population)',
                r'how\s+many\s+students?\s+are\s+there',
            ]
        },
        'student_by_gender': {
            'keywords': ['male', 'female', 'boy', 'girl', 'gender', 'men', 'women'],
            'patterns': [
                r'(?:male|boys?|men|gentlemen)\s+(?:students?|class)',
                r'(?:female|girls?|women|ladies)\s+(?:students?|class)',
                r'students?\s+(?:who are|of|with)\s+(?:gender\s+)?(male|female|boys?|girls?)',
                r'(male|female)\s+students?',
            ]
        },
        'student_by_age': {
            'keywords': ['age', 'years old', 'born in', 'dob', 'older than', 'younger than', 'aged'],
            'patterns': [
                r'students?\s+(?:who are|older than|younger than|of age)\s+(\d+)',
                r'students?\s+aged?\s+(\d+)',
                r'age\s+(?:is|equals?|>=?|<=?)\s*(\d+)',
            ]
        },
        'student_contact': {
            'keywords': ['mobile', 'phone', 'cell', 'landline', 'contact', 'call', 'reach', 'telephone'],
            'patterns': [
                r'(?:contact|mobile|phone|landline)\s+(?:details|info|number)\s+(?:of|for)\s+([a-z][a-z\s]+)',
                r'phone\s+(?:number|no)\s+(?:of|for)\s+([a-z][a-z\s]+)',
                r'mobile\s+number\s+of\s+([a-z][a-z\s]+)',
            ]
        },
        'student_family': {
            'keywords': ['father', 'mother', 'parent', 'guardian', 'family', 'parents'],
            'patterns': [
                r'(?:father|mother|parent|guardian|family)\s+(?:name|details)\s+(?:of|for)\s+([a-z][a-z\s]+)',
                r'parents?\s+of\s+([a-z][a-z\s]+)',
                r'father\s+name\s+of\s+([a-z][a-z\s]+)',
            ]
        },
        'student_address': {
            'keywords': ['address', 'location', 'live', 'residence', 'home', 'ward', 'district', 'province'],
            'patterns': [
                r'(?:address|location|residence)\s+(?:of|for)\s+([a-z][a-z\s]+)',
                r'students?\s+from\s+([a-z][a-z\s]+)',
                r'students?\s+in\s+(?:ward|district|province)\s+([a-z0-9]+)',
            ]
        },
        'student_marks': {
            'keywords': ['marks', 'score', 'grades', 'result', 'performance', 'obtained', 'scored', 'marksheet'],
            'patterns': [
                r'(?:marks|scores|grades|results)\s+(?:of|for|obtained by)\s+([a-z][a-z\s]+)',
                r'(?:subject wise|subject-wise)\s+(?:marks|scores)\s+(?:of|for)\s+([a-z][a-z\s]+)',
                r'show\s+(?:marks|scores)\s+(?:for|of)\s+([a-z][a-z\s]+)',
                r'what\s+(?:marks|scores|grades)\s+did\s+([a-z][a-z\s]+)\s+get',
                r'([a-z][a-z\s]+)\'s\s+(?:marks|scores|grades)',
            ]
        },
        'marks_in_subject': {
            'keywords': ['in subject', 'in', 'for subject', 'of subject', 'subject wise'],
            'patterns': [
                r'(?:marks|scores?|grade)s?\s+(?:in|for|of)\s+([a-z][a-z\s]+?)(?:\s+subject)?\s+(?:of|for|obtained by)\s+([a-z][a-z\s]+)',
                r'([a-z][a-z\s]+)\'s\s+(?:marks|scores?|grades?)\s+(?:in|for)\s+([a-z][a-z\s]+)',
                r'how\s+(?:many|much)\s+did\s+([a-z][a-z\s]+)\s+score\s+in\s+([a-z][a-z\s]+)',
            ]
        },
        'highest_marks': {
            'keywords': ['highest', 'maximum', 'max', 'top', 'best', 'topper', 'highest scorer'],
            'patterns': [
                r'(?:highest|maximum|max|top)\s+(?:marks|score|mark)\s+(?:in|for|of)\s+([a-z][a-z\s]+)',
                r'who\s+(?:scored|got|secured)\s+(?:the\s+)?(?:highest|maximum|top)\s+(?:marks|score)',
                r'topper\s+in\s+([a-z][a-z\s]+)',
            ]
        },
        'lowest_marks': {
            'keywords': ['lowest', 'minimum', 'min', 'bottom', 'worst', 'lowest scorer'],
            'patterns': [
                r'(?:lowest|minimum|min|bottom)\s+(?:marks|score|mark)\s+(?:in|for|of)\s+([a-z][a-z\s]+)',
                r'who\s+(?:scored|got)\s+(?:the\s+)?(?:lowest|minimum|bottom)\s+(?:marks|score)',
            ]
        },
        'average_marks': {
            'keywords': ['average', 'avg', 'mean', 'class average'],
            'patterns': [
                r'(?:average|avg|mean)\s+(?:marks|score|mark)\s+(?:per|for|of)\s+(?:subject|subjects)',
                r'what\s+is\s+(?:the\s+)?(?:average|avg|mean)\s+(?:marks|score)',
            ]
        },
        'total_marks': {
            'keywords': ['total', 'sum', 'aggregate', 'total score'],
            'patterns': [
                r'(?:total|sum|aggregate)\s+(?:marks|score|mark)\s+(?:of|for)\s+([a-z][a-z\s]+)',
                r'(?:total|sum|aggregate)\s+(?:marks|score)',
            ]
        },
        'marks_above_percentage': {
            'keywords': ['above', 'greater than', 'more than', 'over', 'scored above'],
            'patterns': [
                r'scored?\s+above\s+(\d+)%\s+in\s+([a-z][a-z\s]+)',
                r'students?\s+who\s+(?:scored|got)\s+above\s+(\d+)%\s+in\s+([a-z][a-z\s]+)',
            ]
        },
        'marks_below_percentage': {
            'keywords': ['below', 'less than', 'under', 'scored below', 'failed'],
            'patterns': [
                r'scored?\s+below\s+(\d+)%\s+in\s+([a-z][a-z\s]+)',
                r'students?\s+who\s+(?:scored|got)\s+below\s+(\d+)%\s+in\s+([a-z][a-z\s]+)',
                r'students?\s+who\s+failed\s+in\s+([a-z][a-z\s]+)',
            ]
        },
        'gpa_calculation': {
            'keywords': ['gpa', 'grade point average', 'cgpa', 'grade point'],
            'patterns': [
                r'(?:calculate|find|get|show)\s+(?:the\s+)?(?:gpa|grade point average|cgpa)\s+(?:of|for)\s+([a-z][a-z\s]+)',
                r'what\s+is\s+(?:the\s+)?(?:gpa|grade point average)\s+(?:of|for)\s+([a-z][a-z\s]+)',
                r'([a-z][a-z\s]+)\'s\s+(?:gpa|grade point average)',
            ]
        },
        'gpa_distribution': {
            'keywords': ['gpa distribution', 'grade distribution', 'gpa breakdown'],
            'patterns': [
                r'(?:gpa|grade)\s+(?:distribution|analysis|summary|breakdown|statistics)',
                r'how\s+are\s+(?:gpa|grades)\s+(?:distributed|spread)',
                r'students?\s+by\s+(?:gpa|grade)',
            ]
        },
        'gpa_filter': {
            'keywords': ['gpa less than', 'gpa above', 'gpa between', 'students with gpa'],
            'patterns': [
                r'gpa\s+(?:is\s+)?(?:less\s+than|below|under)\s+([A-Z][+-]?|[\d.]+)',
                r'gpa\s+(?:is\s+)?(?:greater\s+than|above|over)\s+([A-Z][+-]?|[\d.]+)',
                r'gpa\s+(?:is\s+)?between\s+([A-Z][+-]?|[\d.]+)\s+(?:and|to)\s+([A-Z][+-]?|[\d.]+)',
            ]
        },
        'grade_analysis': {
            'keywords': ['grade', 'A+', 'A', 'B+', 'B', 'C+', 'C', 'D', 'NG', 'grading'],
            'patterns': [
                r'students?\s+with\s+grade\s+([A-Z][+-]?)',
                r'(?:how many|count)\s+students?\s+got\s+([A-Z][+-]?)',
                r'grade\s+([A-Z][+-]?)\s+(?:students|class)',
            ]
        },
        'pass_fail_analysis': {
            'keywords': ['pass', 'fail', 'passed', 'failed', 'pass rate', 'fail rate', 'pass percentage'],
            'patterns': [
                r'(?:pass|fail)\s+(?:percentage|rate|ratio)\s+(?:of|for)\s+([a-z][a-z\s]+)',
                r'pass\s+(?:percentage|rate)\s+(?:of|for)\s+([a-z][a-z\s]+)',
                r'fail\s+(?:percentage|rate)\s+(?:of|for)\s+([a-z][a-z\s]+)',
                r'(?:how many|count)\s+students?\s+passed\s+in\s+([a-z][a-z\s]+)',
                r'(?:how many|count)\s+students?\s+failed\s+in\s+([a-z][a-z\s]+)',
            ]
        },
        'failed_students': {
            'keywords': ['failed', 'fail', 'failed students', 'fail list', 'failing'],
            'patterns': [
                r'(?:which|who|list)\s+(?:students|who)\s+(?:have\s+)?failed\s+(?:in|on)\s+([a-z][a-z\s]+)',
                r'students\s+who\s+failed',
                r'failed\s+students',
            ]
        },
        'passed_students': {
            'keywords': ['passed', 'pass', 'passed students', 'pass list', 'successful'],
            'patterns': [
                r'(?:which|who|list)\s+(?:students|who)\s+(?:have\s+)?passed\s+(?:in|on)\s+([a-z][a-z\s]+)',
                r'students\s+who\s+passed',
                r'passed\s+students',
            ]
        },
        'attendance': {
            'keywords': ['attendance', 'present', 'absent', 'regular', 'attendance report'],
            'patterns': [
                r'(?:attendance|present)\s+(?:of|for)\s+([a-z][a-z\s]+)',
                r'attendance\s+(?:report|summary|record|details)',
                r'students?\s+with\s+(?:low|poor|bad)\s+attendance',
                r'students?\s+with\s+(?:high|good|regular)\s+attendance',
            ]
        },
        'attendance_above': {
            'keywords': ['attendance above', 'attendance greater than', 'attendance over', 'high attendance'],
            'patterns': [
                r'attendance\s+(?:above|greater than|over|>=?)\s+(\d+)%',
                r'students?\s+with\s+attendance\s+(?:above|greater than|over|>=?)\s+(\d+)%',
            ]
        },
        'attendance_below': {
            'keywords': ['attendance below', 'attendance less than', 'attendance under', 'poor attendance', 'low attendance'],
            'patterns': [
                r'attendance\s+(?:below|less than|under|<=?)\s+(\d+)%',
                r'students?\s+with\s+attendance\s+(?:below|less than|under|<=?)\s+(\d+)%',
                r'(?:low|poor|bad)\s+attendance',
            ]
        },
        'top_students': {
            'keywords': ['top', 'best', 'highest', 'topper', 'merit', 'honor roll'],
            'patterns': [
                r'(?:top|best|highest)\s+(\d+)?\s*(?:students|performers)',
                r'topper\s+(?:of|in)\s+(?:the\s+)?(?:class|section)',
                r'merit\s+(?:list|students)',
                r'students?\s+with\s+(?:distinction|highest marks)',
            ]
        },
        'bottom_students': {
            'keywords': ['bottom', 'worst', 'lowest', 'struggling'],
            'patterns': [
                r'(?:bottom|worst|lowest)\s+(\d+)?\s*(?:students|performers)',
                r'students?\s+(?:at|in)\s+(?:the\s+)?(?:bottom|last)',
                r'lowest\s+performing\s+students',
            ]
        },
        'rank': {
            'keywords': ['rank', 'ranking', 'position', 'class rank', 'overall rank'],
            'patterns': [
                r'rank\s+(?:of|for)\s+([a-z][a-z\s]+)',
                r'ranking\s+(?:of|for)\s+([a-z][a-z\s]+)',
                r'([a-z][a-z\s]+)\'s\s+(?:rank|position)',
                r'rank\s+list',
            ]
        },
        'list_subjects': {
            'keywords': ['subjects', 'courses', 'subject list', 'available subjects'],
            'patterns': [
                r'(?:list|show|get|display)\s+(?:all\s+)?(?:subjects|subject|courses)',
                r'(?:how many|number of)\s+(?:subjects|subject|courses)',
                r'what\s+subjects\s+are\s+there',
            ]
        },
        'subject_details': {
            'keywords': ['subject details', 'subject info', 'subject code', 'credit hours'],
            'patterns': [
                r'subject\s+(?:details|info|information)\s+(?:for|of)\s+([a-z][a-z\s]+)',
                r'what\s+is\s+([a-z][a-z\s]+)\s+subject',
                r'([a-z][a-z\s]+)\s+(?:code|credit|type)',
            ]
        },
        'subject_performance': {
            'keywords': ['subject performance', 'performance in subject', 'subject analysis'],
            'patterns': [
                r'(?:performance|analysis)\s+(?:of|in)\s+([a-z][a-z\s]+)\s+(?:subject|class)',
                r'([a-z][a-z\s]+)\s+subject\s+(?:performance|analysis)',
                r'subject\s+wise\s+(?:performance|analysis)',
            ]
        },
        'list_exams': {
            'keywords': ['exams', 'tests', 'examinations', 'terminal', 'exam list'],
            'patterns': [
                r'(?:list|show|get|display)\s+(?:all\s+)?(?:exams|examinations|tests)',
                r'(?:how many|number of)\s+(?:exams|examinations|tests)',
                r'exam\s+(?:schedule|dates|details)',
            ]
        },
        'exam_details': {
            'keywords': ['exam details', 'exam info', 'exam weightage', 'exam type'],
            'patterns': [
                r'exam\s+(?:details|info|information)\s+(?:for|of)\s+([a-z][a-z\s]+)',
                r'what\s+is\s+([a-z][a-z\s]+)\s+exam',
                r'([a-z][a-z\s]+)\s+exam\s+(?:weightage|type|date)',
            ]
        },
        'section_students': {
            'keywords': ['section', 'class section', 'section list', 'section A/B/C'],
            'patterns': [
                r'students\s+(?:in|of|from)\s+section\s+([A-Z])',
                r'section\s+([A-Z])\s+students',
                r'class\s+([A-Z])\s+section',
                r'who\s+is\s+in\s+section\s+([A-Z])',
            ]
        },
        'section_strength': {
            'keywords': ['section strength', 'section size', 'class strength', 'students per section'],
            'patterns': [
                r'(?:class|section)\s+(?:strength|size|count|population)',
                r'students\s+per\s+section',
                r'how\s+many\s+students\s+in\s+section',
            ]
        },
        'ethnicity_distribution': {
            'keywords': ['ethnicity', 'ethnic', 'caste', 'tribe', 'ethnic distribution'],
            'patterns': [
                r'(?:ethnicity|ethnic|caste|tribe)\s+(?:distribution|analysis|summary|breakdown|statistics)',
                r'students\s+by\s+(?:ethnicity|ethnic|caste)',
                r'show\s+ethnicity\s+distribution',
            ]
        },
        'religion_distribution': {
            'keywords': ['religion', 'religious', 'faith', 'religion distribution'],
            'patterns': [
                r'(?:religion|religious|faith)\s+(?:distribution|analysis|summary|breakdown|statistics)',
                r'students\s+by\s+(?:religion|religious)',
                r'show\s+religion\s+distribution',
            ]
        },
        'blood_group_distribution': {
            'keywords': ['blood group', 'blood type', 'blood group distribution'],
            'patterns': [
                r'(?:blood\s+group|blood\s+type|blood)\s+(?:distribution|analysis|summary|breakdown|statistics)',
                r'students\s+by\s+(?:blood\s+group|blood\s+type)',
                r'show\s+blood\s+group\s+distribution',
            ]
        },
        'gender_distribution': {
            'keywords': ['gender distribution', 'gender ratio', 'sex ratio', 'male female ratio'],
            'patterns': [
                r'(?:gender|sex)\s+(?:distribution|analysis|summary|breakdown|statistics|ratio|balance)',
                r'students\s+by\s+(?:gender|sex)',
                r'male\s+to\s+female\s+ratio',
                r'gender\s+ratio',
            ]
        },
        'age_distribution': {
            'keywords': ['age distribution', 'age analysis', 'students by age', 'age group'],
            'patterns': [
                r'(?:age|dob)\s+(?:distribution|analysis|summary|breakdown|statistics)',
                r'students\s+by\s+(?:age|age group)',
                r'age\s+(?:group|range)',
            ]
        },
        'address_distribution': {
            'keywords': ['address', 'location', 'district', 'province', 'ward'],
            'patterns': [
                r'(?:address|location)\s+(?:distribution|analysis|summary)',
                r'students\s+by\s+(?:district|province|ward|city|town)',
                r'(?:district|province|ward)\s+(?:distribution|count|statistics)',
            ]
        },
        'compare_students': {
            'keywords': ['compare', 'comparison', 'vs', 'versus', 'between', 'against'],
            'patterns': [
                r'compare\s+(?:marks|performance)\s+of\s+([^,]+?)\s*(?:,|and)\s*([^,]+?)(?:\s*(?:,|and)\s*([^,]+?))?',
                r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:vs|versus|and|with)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
                r'compare\s+([^,]+?)\s*(?:,)\s*([^,]+?)(?:\s*(?:,|and)\s*([^,]+?))?',
                r'who\s+is\s+(?:better|best)\s+(?:between|among)\s+([^,]+?)\s+(?:and|,)\s*([^,]+?)(?:\s*(?:,|and)\s*([^,]+?))?',
                r'difference\s+between\s+([^,]+?)\s+(?:and|,)\s*([^,]+?)',
                r'compare\s+([^,]+?)\s+(?:vs|versus)\s+([^,]+?)',
                r'([^,]+?)\s+vs\s+([^,]+?)\s+marks',
            ]
        },
        'statistics': {
            'keywords': ['statistics', 'stats', 'data analysis', 'overview', 'class stats'],
            'patterns': [
                r'(?:statistics|stats)\s+(?:summary|analysis)',
                r'class\s+statistics',
                r'show\s+statistics',
            ]
        },
        'passout_students': {
            'keywords': ['passout', 'graduated', 'passed out', 'graduate', 'alumni', 'class 10'],
            'patterns': [
                r'(?:passout|graduated|passed out|graduate)\s+(?:students|student)',
                r'students\s+who\s+(?:passed|graduated|completed)',
                r'class\s+10\s+students',
            ]
        },
        'academic_report': {
            'keywords': ['report card', 'progress report', 'academic report', 'performance report', 'transcript'],
            'patterns': [
                r'(?:academic|performance|progress)\s+(?:report|card|summary)\s+(?:of|for)\s+([a-z][a-z\s]+)',
                r'(?:report card|progress card|transcript)\s+(?:of|for)\s+([a-z][a-z\s]+)',
                r'show\s+report\s+card\s+of\s+([a-z][a-z\s]+)',
            ]
        },
        'performance_summary': {
            'keywords': ['performance summary', 'performance analysis', 'academic standing', 'performance overview'],
            'patterns': [
                r'(?:performance|academic)\s+(?:summary|analysis|overview)',
                r'summary\s+of\s+(?:performance|academic\s+performance)',
                r'overall\s+performance\s+summary',
            ]
        },
        'complex_filter': {
            'keywords': ['and', 'or', 'with', 'having', 'multiple conditions', 'complex query', 'filter'],
            'patterns': [
                r'students?\s+with\s+(\w+)\s+(?:above|greater than|over|>=?)\s+([\d.]+)\s+and\s+(\w+)\s+(?:above|greater than|over|>=?)\s+([\d.]+)',
                r'students?\s+with\s+(\w+)\s+(?:below|less than|under|<=?)\s+([\d.]+)\s+and\s+(\w+)\s+(?:below|less than|under|<=?)\s+([\d.]+)',
                r'students?\s+who\s+(?:have|are)\s+(.+?)\s+and\s+(.+?)\s+and\s+(.+)',
                r'students?\s+with\s+gpa\s+above\s+([\d.]+)\s+and\s+attendance\s+above\s+(\d+)%',
            ]
        }
    }

    @classmethod
    def parse(cls, query: str) -> Dict[str, Any]:
        query_lower = query.lower().strip()
        query_clean = re.sub(r'\?+$', '', query_lower).strip()

        if not query_clean:
            return {'query_type': 'help', 'entities': {}, 'original_query': query, 'confidence': 1.0}

        if any(w in query_clean for w in ['help', 'assist', 'support', 'guide', 'what can you do',
                                           'capabilities', 'features', 'hello', 'hi', 'hey']):
            return {'query_type': 'help', 'entities': {}, 'original_query': query, 'confidence': 1.0}

        multi_students = cls._parse_multi_student_comparison(query_clean)
        if multi_students and len(multi_students) >= 2:
            return {
                'query_type': 'compare_students',
                'entities': {'students': multi_students, 'student_count': len(multi_students)},
                'original_query': query,
                'confidence': 0.98
            }

        result = {'query_type': 'general', 'entities': {}, 'original_query': query,
                   'confidence': 0.0, 'matched_keywords': []}

        for query_type, mapping in cls.SEMANTIC_MAPPINGS.items():
            for pattern in mapping.get('patterns', []):
                match = re.search(pattern, query_clean, re.IGNORECASE)
                if match:
                    result['query_type'] = query_type
                    result['confidence'] = 0.95
                    groups = match.groups()
                    if groups:
                        result['entities'] = cls._extract_entities_from_match(query_type, groups, query_clean)
                    return result

        for query_type, mapping in cls.SEMANTIC_MAPPINGS.items():
            if cls._matches_keywords(query_clean, mapping.get('keywords', [])):
                result['query_type'] = query_type
                result['matched_keywords'] = cls._get_matched_keywords(query_clean, mapping.get('keywords', []))
                result['confidence'] = 0.7
                result['entities'] = cls._extract_entities_generic(query_clean)
                return result

        student_name = cls._extract_student_name_generic(query_clean)
        if student_name:
            result['query_type'] = 'student_details'
            result['entities']['student_name'] = student_name
            result['confidence'] = 0.5

        return result

    STOPWORD_SUFFIX = re.compile(r'\s+(please|now|today|thanks?|pls)\W*$', re.IGNORECASE)

    @classmethod
    def _clean_captured_name(cls, raw: str) -> str:
        return cls.STOPWORD_SUFFIX.sub('', raw.strip()).strip()

    @classmethod
    def _parse_multi_student_comparison(cls, query: str) -> Optional[List[str]]:
        patterns = [
            r'compare\s+(?:marks|performance|subject\s*wise\s*marks)\s+of\s+([^,]+?)\s*(?:,|and)\s*([^,]+?)(?:\s*(?:,|and)\s*([^,]+?))?',
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:vs|versus|and|with)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'compare\s+([^,]+?)\s*(?:,)\s*([^,]+?)\s*(?:,)\s*([^,]+?)',
            r'([^,]+?)\s+vs\s+([^,]+?)\s+marks',
            r'compare\s+marks\s+of\s+([^,]+?)\s+(?:and|vs|versus)\s+([^,]+?)',
        ]

        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                students = []
                for g in match.groups():
                    if g and g.strip():
                        name = g.strip()
                        name = re.sub(r'\s*(?:marks|subject|wise|of|the|performance|results)\s*$', '', name).strip()
                        name = cls._clean_captured_name(name)
                        if name and len(name) > 2:
                            students.append(name)
                if len(students) >= 2:
                    return students
        return None

    @classmethod
    def _matches_keywords(cls, query: str, keywords: List[str]) -> bool:
        query_lower = query.lower()
        matches = 0
        for keyword in keywords:
            if keyword in query_lower:
                matches += 1
            if matches >= 2:
                return True
        return False

    @classmethod
    def _get_matched_keywords(cls, query: str, keywords: List[str]) -> List[str]:
        query_lower = query.lower()
        return [kw for kw in keywords if kw in query_lower]

    @classmethod
    def _extract_entities_from_match(cls, query_type: str, groups: Tuple, query: str) -> Dict:
        entities = {}

        if query_type in ['student_details', 'student_marks', 'gpa_calculation', 'attendance',
                           'student_contact', 'student_family', 'student_address', 'academic_report']:
            if groups and groups[0]:
                entities['student_name'] = cls._clean_captured_name(groups[0])

        elif query_type == 'marks_in_subject':
            if len(groups) >= 2:
                entities['subject'] = groups[0].strip()
                entities['student_name'] = cls._clean_captured_name(groups[1])

        elif query_type in ['highest_marks', 'lowest_marks', 'subject_performance', 'subject_details']:
            if groups and groups[0]:
                entities['subject'] = groups[0].strip()

        elif query_type in ['marks_above_percentage', 'marks_below_percentage']:
            if len(groups) >= 2:
                try:
                    entities['percentage'] = float(groups[0])
                except Exception:
                    pass
                entities['subject'] = groups[1].strip()

        elif query_type == 'student_by_age':
            if groups and groups[0]:
                try:
                    entities['age'] = int(groups[0])
                except Exception:
                    pass

        elif query_type == 'student_by_gender':
            if groups and groups[0]:
                gender = groups[0].strip().lower()
                if gender in ['male', 'boy', 'boys', 'men', 'gentlemen']:
                    entities['gender'] = 'male'
                elif gender in ['female', 'girl', 'girls', 'women', 'ladies']:
                    entities['gender'] = 'female'
                else:
                    entities['gender'] = gender

        elif query_type in ['section_students', 'section_strength']:
            if groups and groups[0]:
                entities['section'] = groups[0].strip().upper()

        elif query_type in ['top_students', 'bottom_students']:
            if groups and groups[0] and groups[0].isdigit():
                entities['limit'] = int(groups[0])
            else:
                entities['limit'] = 10

        elif query_type == 'gpa_filter':
            if len(groups) >= 1:
                try:
                    if groups[0].strip().upper() in GRADE_TO_GPA_MAP:
                        entities['grade'] = groups[0].strip().upper()
                    else:
                        entities['gpa_threshold'] = float(groups[0])
                except Exception:
                    pass
            if len(groups) >= 2:
                try:
                    entities['gpa_threshold2'] = float(groups[1])
                except Exception:
                    pass

        elif query_type == 'grade_analysis':
            if groups and groups[0]:
                entities['grade'] = groups[0].strip().upper()

        elif query_type == 'rank':
            if groups and groups[0]:
                entities['student_name'] = cls._clean_captured_name(groups[0])

        elif query_type == 'exam_details':
            if groups and groups[0]:
                entities['exam_name'] = groups[0].strip()

        elif query_type in ['attendance_above', 'attendance_below']:
            if groups and groups[0]:
                try:
                    entities['attendance_threshold'] = float(groups[0])
                except Exception:
                    pass

        return entities

    @classmethod
    def _extract_entities_generic(cls, query: str) -> Dict:
        entities = {}

        name = cls._extract_student_name_generic(query)
        if name:
            entities['student_name'] = name

        subject = cls._extract_subject_generic(query)
        if subject:
            entities['subject'] = subject

        gender_match = re.search(r'\b(male|female|boys?|girls?|men|women)\b', query.lower())
        if gender_match:
            entities['gender'] = gender_match.group(1)

        num_match = re.search(r'\b(\d+)\b', query)
        if num_match:
            entities['number'] = int(num_match.group(1))

        section_match = re.search(r'section\s+([A-Z])', query, re.IGNORECASE)
        if section_match:
            entities['section'] = section_match.group(1).upper()

        grade_match = re.search(r'\b([A-D][+-]?|NG)\b', query.upper())
        if grade_match:
            entities['grade'] = grade_match.group(1)

        return entities

    @classmethod
    def _extract_student_name_generic(cls, query: str) -> Optional[str]:
        patterns = [
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b',
            r'(?:of|for|about|from)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})',
            r'(?:is|are|who)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})',
        ]

        for pattern in patterns:
            match = re.search(pattern, query)
            if match:
                name = match.group(1).strip()
                if name.lower() not in ['Student', 'Teacher', 'Class', 'Section', 'Subject',
                                         'English', 'Nepali', 'Math', 'Science', 'Social']:
                    return cls._clean_captured_name(name)
        return None

    @classmethod
    def _extract_subject_generic(cls, query: str) -> Optional[str]:
        subject_patterns = [
            r'\b(english|nepali|math|mathematics|science|social|physics|chemistry|biology|computer|population|economics)\b',
            r'(?:subject|subject name|in|for)\s+([a-z][a-z\s]+?)(?:\s+subject)?',
        ]
        for pattern in subject_patterns:
            match = re.search(pattern, query.lower(), re.IGNORECASE)
            if match:
                subject = match.group(1) if len(match.groups()) > 0 else match.group(0)
                return subject.strip().title()
        return None


# =============================================================================
# MULTI-STUDENT SQL GENERATOR  (resolver-based, no cartesian self-join)
# =============================================================================

class MultiStudentSQLGenerator:
    @staticmethod
    def generate(sql_gen: 'SQLGenerator', students: List[str]) -> Tuple[Optional[str], Optional[List[str]], List[str]]:
        """
        Resolves each name to a uuid first, then builds one uniform join keyed
        on those known IDs  works for 2..N students with no self-join blowup.

        Returns (sql, resolved_display_names, not_found_names).
        If any name fails to resolve, sql and resolved_display_names are None
        and not_found_names lists every name that couldn't be matched.
        """
        resolved = []
        not_found = []
        for name in students:
            hit = StudentResolver.resolve_uuid(sql_gen, name)
            if hit is None:
                not_found.append(name)
            else:
                resolved.append(hit)

        if not_found:
            return None, None, not_found

        uuids = [r[0] for r in resolved]
        display_names = [r[1] for r in resolved]
        clean_names = [n.replace(' ', '_') for n in display_names]

        select_parts = ["sub.title AS subject", "ep.fullMark AS full_marks"]
        from_parts = [
            "exam_parameters ep",
            "JOIN subjects sub ON ep.class_section_subject_id = sub.cssId",
        ]
        where_parts = []

        for i, (uid, clean) in enumerate(zip(uuids, clean_names), start=1):
            select_parts.append(f"m{i}.mark AS {clean}_mark")
            select_parts.append(
                f"ROUND(CAST(m{i}.mark AS REAL) * 100.0 / CAST(ep.fullMark AS REAL), 2) AS {clean}_percentage"
            )
            select_parts.append(
                f"GET_GRADE(CAST(m{i}.mark AS REAL) * 100.0 / CAST(ep.fullMark AS REAL)) AS {clean}_grade"
            )
            from_parts.append(
                f"LEFT JOIN marks m{i} ON m{i}.exam_parameter_id = ep.id AND m{i}.student_id = '{uid}'"
            )
            where_parts.append(f"m{i}.mark IS NOT NULL")

        sql = f"""
        SELECT {', '.join(select_parts)}
        FROM {' '.join(from_parts)}
        WHERE {' OR '.join(where_parts)}
        ORDER BY sub.title
        """
        return sql, display_names, []


# =============================================================================
# MULTI-STUDENT FORMATTER
# =============================================================================

class MultiStudentFormatter:
    @staticmethod
    def format(df: pd.DataFrame, students: List[str]) -> Dict[str, str]:
        if df.empty:
            return {
                'formatted': 'No data found for the comparison. Please check the student names.',
                'interpretation': '**1. Overview**\n\nNo data matches your query. Please check the student names or filter conditions.'
            }

        lines = []

        lines.append("**1. Multi-Student Comparison Overview**")
        lines.append("")
        lines.append(f"Comparing **{len(students)}** students: **{', '.join(students)}**")
        lines.append(f"Based on **{len(df)}** subjects with available data.")
        lines.append("")

        lines.append("**2. Subject-wise Performance Comparison**")
        lines.append("")

        student_totals = {student: {'obtained': 0, 'full': 0} for student in students}

        for _, row in df.iterrows():
            subject = row.get('subject', 'Unknown')
            full_marks = row.get('full_marks', 0)

            lines.append(f"**{subject}** (Full Marks: {full_marks}):")
            for student in students:
                clean_name = student.replace(' ', '_')
                mark_col = f"{clean_name}_mark"
                pct_col = f"{clean_name}_percentage"
                grade_col = f"{clean_name}_grade"

                mark = row.get(mark_col, 'N/A')
                pct = row.get(pct_col, 'N/A')
                grade = row.get(grade_col, 'N/A')

                if pct != 'N/A' and not pd.isna(pct) and mark != 'N/A' and not pd.isna(mark):
                    lines.append(f"  - {student}: {mark:.1f}/{full_marks:.1f} ({pct:.1f}%) - Grade: {grade}")
                    student_totals[student]['obtained'] += mark
                    student_totals[student]['full'] += full_marks
                else:
                    lines.append(f"  - {student}: No data available")
            lines.append("")

        lines.append("**3. Individual Performance Summary**")
        lines.append("")

        for student in students:
            clean_name = student.replace(' ', '_')
            pct_col = f"{clean_name}_percentage"

            if pct_col in df.columns:
                pct_data = df[pct_col].dropna()
                if len(pct_data) > 0:
                    avg_pct = pct_data.mean()
                    min_pct = pct_data.min()
                    max_pct = pct_data.max()

                    total_obtained = student_totals[student]['obtained']
                    total_full = student_totals[student]['full']
                    overall_pct = (total_obtained / total_full * 100) if total_full > 0 else 0
                    overall_grade = get_grade_from_percentage(overall_pct)
                    overall_gpa = get_gpa_from_percentage(overall_pct)

                    lines.append(f"**{student}**:")
                    lines.append(f"  - Average: {avg_pct:.1f}%")
                    lines.append(f"  - Range: {min_pct:.1f}% to {max_pct:.1f}%")
                    lines.append(f"  - Overall Percentage: {overall_pct:.1f}%")
                    lines.append(f"  - Overall Grade: {overall_grade}")
                    lines.append(f"  - Overall GPA: {overall_gpa:.2f}")
                    lines.append(f"  - Total Marks: {total_obtained:.1f} out of {total_full:.1f}")

                    if 'subject' in df.columns and pct_col in df.columns and not df[pct_col].isna().all():
                        max_row = df.loc[df[pct_col].idxmax()]
                        min_row = df.loc[df[pct_col].idxmin()]
                        lines.append(f"  - Best: {max_row.get('subject', 'Unknown')} ({max_pct:.1f}%)")
                        lines.append(f"  - Weakest: {min_row.get('subject', 'Unknown')} ({min_pct:.1f}%)")
                    lines.append("")

        lines.append("**4. Subject-wise Winners**")
        lines.append("")

        wins = {student: 0 for student in students}
        for _, row in df.iterrows():
            subject = row.get('subject', 'Unknown')
            scores = {}
            for student in students:
                clean_name = student.replace(' ', '_')
                pct_col = f"{clean_name}_percentage"
                if pct_col in df.columns and not pd.isna(row[pct_col]):
                    scores[student] = row[pct_col]

            if scores:
                winner = max(scores.items(), key=lambda x: x[1])
                wins[winner[0]] = wins.get(winner[0], 0) + 1
                lines.append(f"- **{subject}**: {winner[0]} ({winner[1]:.1f}%)")

        lines.append("")
        lines.append("**Win Count Summary:**")
        for student, count in sorted(wins.items(), key=lambda x: x[1], reverse=True):
            if count > 0:
                lines.append(f"- {student}: {count} subject{'s' if count > 1 else ''}")
        lines.append("")

        lines.append("**5. Recommendations**")
        lines.append("")

        has_recommendations = False
        for student in students:
            clean_name = student.replace(' ', '_')
            pct_col = f"{clean_name}_percentage"
            if pct_col in df.columns:
                weak_rows = df[df[pct_col].notna()]
                if not weak_rows.empty:
                    weak_row = weak_rows.loc[weak_rows[pct_col].idxmin()]
                    weak_subject = weak_row.get('subject', 'Unknown')
                    weak_pct = weak_row[pct_col]

                    if weak_pct < 60:
                        gap = 60 - weak_pct
                        lines.append(
                            f"- **{student}**: {weak_subject} is **{gap:.1f} points** below the 60% mark "
                            f"({weak_pct:.1f}%)  closest area to focus on."
                        )
                        has_recommendations = True
                    elif weak_pct < 75:
                        gap = 75 - weak_pct
                        lines.append(
                            f"- **{student}**: {weak_subject} at **{weak_pct:.1f}%** is {gap:.1f} points "
                            f"short of the 75% distinction mark."
                        )
                        has_recommendations = True

        if not has_recommendations:
            lines.append("Every student is comfortably clear of the 75% mark in their weakest subject.")

        formatted = "\n".join(lines)
        interpretation = re.sub(r'^\d+\.\s+', '', formatted, flags=re.MULTILINE)

        return {'formatted': formatted, 'interpretation': interpretation}


# =============================================================================
# RESULT FORMATTER  (dynamic interpretation  no canned per-bracket sentences)
# =============================================================================

class ResultFormatter:

    STRENGTH_THRESHOLD = 60
    WEAKNESS_THRESHOLD = 60  # single shared cutoff, no gap between the two sections

    DISTRIBUTION_TYPES = {
        'gpa_distribution', 'grade_analysis', 'ethnicity_distribution',
        'religion_distribution', 'blood_group_distribution',
        'gender_distribution', 'age_distribution', 'address_distribution',
        'pass_fail_analysis',
    }
    RANKED_LIST_TYPES = {
        'top_students', 'bottom_students', 'rank', 'performance_summary',
    }
    SINGLE_METRIC_TYPES = {
        'gpa_calculation', 'total_marks', 'attendance',
    }

    @classmethod
    def format(cls, query: str, result_df: pd.DataFrame, query_type: str,
               entities: Dict, sql: str = None, students: List[str] = None) -> Dict[str, Any]:

        if result_df.empty:
            return {
                "formatted": "No results found.",
                "interpretation": "**1. Overview**\n\nNo data matches your query. Please check the filter conditions or data availability.",
                "sql": sql,
                "rows": 0,
                "data": None
            }

        if query_type == 'compare_students' and students and len(students) >= 2:
            formatted = MultiStudentFormatter.format(result_df, students)
            return {
                "formatted": formatted['formatted'],
                "interpretation": formatted['interpretation'],
                "sql": sql,
                "rows": len(result_df),
                "data": convert_to_serializable(result_df)
            }

        if query_type == 'student_marks':
            return cls._format_student_marks(query, result_df, entities, sql)

        if query_type in cls.DISTRIBUTION_TYPES:
            return cls._format_distribution(query, result_df, query_type, entities, sql)

        if query_type in cls.RANKED_LIST_TYPES:
            return cls._format_ranked_list(query, result_df, query_type, entities, sql)

        if query_type in cls.SINGLE_METRIC_TYPES:
            return cls._format_single_metric(query, result_df, query_type, entities, sql)

        return cls._format_generic(query, result_df, query_type, entities, sql)

    # ------------------------------------------------------------------
    # Dynamic recommendation generator  every line is built from the
    # actual computed numbers for THIS student. No fixed bracket text.
    # ------------------------------------------------------------------
    @classmethod
    def _generate_dynamic_recommendation(
        cls,
        student_name: str,
        overall_pct: float,
        subject_data: List[Tuple[str, float, float, float, str]],
    ) -> List[str]:
        """
        subject_data: list of (subject, obtained, full, percentage, grade)
        Returns markdown lines built entirely from real numbers  current
        grade band + exact points to the next band, the actual weak
        subjects with actual point-gaps, the actual strongest subject,
        and the actual marks needed in the weakest subject to cross 60%.
        """
        lines: List[str] = []

        if not subject_data:
            return ["- No recommendation available: no subject data to analyze."]

        sorted_by_pct = sorted(subject_data, key=lambda x: x[3])
        weakest = sorted_by_pct[0]
        strongest = sorted_by_pct[-1]

        # 1. Current grade band + exact distance to the next band up
        current_band = get_grade_band(overall_pct)
        next_band = get_next_grade_band(current_band)
        if next_band:
            points_needed = next_band["min_percentage"] - overall_pct
            lines.append(
                f"- At **{overall_pct:.2f}%** (grade **{current_band['grade']}**), "
                f"**{points_needed:.2f} more percentage points** would reach grade "
                f"**{next_band['grade']}** ({next_band['min_percentage']}%+)."
            )
        else:
            lines.append(
                f"- At **{overall_pct:.2f}%**, already in the top grade band (**{current_band['grade']}**)."
            )

        # 2. Actual subjects below 60%, named, with actual point-gap
        below_60 = [s for s in sorted_by_pct if s[3] < 60]
        if below_60:
            detail = "; ".join(
                f"**{name}** ({pct:.1f}%, {60 - pct:.1f} pts below 60%)"
                for name, obtained, full, pct, grade in below_60[:3]
            )
            extra = f" and {len(below_60) - 3} more" if len(below_60) > 3 else ""
            lines.append(f"- **{len(below_60)}** subject(s) sit below 60%: {detail}{extra}.")
        else:
            lines.append(
                f"- Every subject is at or above 60% , the lowest is **{weakest[0]}** at **{weakest[3]:.1f}%**."
            )

        # 3. Strongest subject, named, with the actual gap over the weakest
        gap = strongest[3] - weakest[3]
        lines.append(
            f"- Strongest subject is **{strongest[0]}** at **{strongest[3]:.1f}%** (grade **{strongest[4]}**)  "
            f"a **{gap:.1f}-point gap** over the weakest subject, **{weakest[0]}** ({weakest[3]:.1f}%)."
        )

        # 4. Concrete, actionable target: exact marks needed in the weakest
        #    subject alone to cross the 60% line.
        w_name, w_obtained, w_full, w_pct, w_grade = weakest
        target_obtained = 0.6 * w_full
        if target_obtained > w_obtained:
            marks_needed = target_obtained - w_obtained
            lines.append(
                f"- Raising **{w_name}** by **{marks_needed:.1f} marks** "
                f"(to {target_obtained:.1f}/{w_full:.1f}) would bring that subject to the 60% mark."
            )

        # 5. Count of subjects deep in trouble (<40%), named  only if any exist
        critical = [s for s in sorted_by_pct if s[3] < 40]
        if critical:
            names = ", ".join(f"**{s[0]}**" for s in critical)
            lines.append(
                f"- {len(critical)} subject(s) are below 40% and need the most immediate attention: {names}."
            )

        return lines

    # ------------------------------------------------------------------
    # student_marks full narrative, dynamic recommendations
    # ------------------------------------------------------------------
    @classmethod
    def _format_student_marks(cls, query: str, result_df: pd.DataFrame, entities: Dict, sql: str) -> Dict[str, Any]:
        student_name = entities.get('student_name', 'Student')
        lines = []
        lines.append(f"**1. Marks Summary for {student_name}**")
        lines.append("")
        lines.append(f"This query shows the marks for **{student_name}** across **{len(result_df)} subjects**.")
        lines.append("")

        lines.append("**2. Subject-wise Performance**")
        lines.append("")

        total_obtained = 0.0
        total_full = 0.0
        subject_data: List[Tuple[str, float, float, float, str]] = []

        for _, row in result_df.iterrows():
            subject = row.get('subject', 'Unknown')
            obtained = row.get('obtained_mark')
            full = row.get('full_marks')
            if obtained is not None and full is not None and full > 0:
                try:
                    obtained = float(obtained)
                    full = float(full)
                    pct = (obtained / full * 100)
                    grade = get_grade_from_percentage(pct)
                    total_obtained += obtained
                    total_full += full
                    subject_data.append((subject, obtained, full, pct, grade))
                    lines.append(f"- **{subject}**: {obtained:.1f} out of {full:.1f} ({pct:.1f}%) - Grade: **{grade}**")
                except Exception:
                    lines.append(f"- **{subject}**: Data unavailable")
        lines.append("")

        sorted_data = sorted(subject_data, key=lambda x: x[3], reverse=True) if subject_data else []

        lines.append("**3. Areas of Strength**")
        lines.append("")
        strengths = [s for s in sorted_data if s[3] >= cls.STRENGTH_THRESHOLD]
        if strengths:
            for subject, obtained, full, pct, grade in strengths:
                lines.append(f"- **{subject}**: {obtained:.1f} out of {full:.1f} ({pct:.1f}%) - Grade: **{grade}**")
        elif sorted_data:
            lines.append(f"No subjects scoring at or above {cls.STRENGTH_THRESHOLD} percent. The best performing subjects are:")
            for subject, obtained, full, pct, grade in sorted_data[:2]:
                lines.append(f"- **{subject}**: {obtained:.1f} out of {full:.1f} ({pct:.1f}%) - Grade: **{grade}**")
        else:
            lines.append("- No subject data available for strength analysis")
        lines.append("")

        lines.append("**4. Areas for Improvement**")
        lines.append("")
        weaknesses = [s for s in sorted_data if s[3] < cls.WEAKNESS_THRESHOLD]
        if weaknesses:
            for subject, obtained, full, pct, grade in weaknesses:
                lines.append(f"- **{subject}**: {obtained:.1f} out of {full:.1f} ({pct:.1f}%) - Grade: **{grade}**")
        elif sorted_data:
            lines.append(f"- All subjects scoring at or above {cls.WEAKNESS_THRESHOLD} percent")
        else:
            lines.append("- No subject data available for improvement analysis")
        lines.append("")

        lines.append("**5. Overall Summary (NEB Grading)**")
        lines.append("")
        overall_pct = 0.0
        if total_full > 0:
            overall_pct = (total_obtained / total_full * 100)
            overall_grade = get_grade_from_percentage(overall_pct)
            overall_gpa = get_gpa_from_percentage(overall_pct)
            lines.append(f"- Overall Percentage: **{overall_pct:.2f}%**")
            lines.append(f"- Overall GPA: **{overall_gpa:.2f}**")
            lines.append(f"- Overall Grade: **{overall_grade}**")
            lines.append(f"- Total Marks: **{total_obtained:.1f}** out of **{total_full:.1f}**")
        else:
            lines.append("- No overall summary available")
        lines.append("")

        # ---- Recommendations: fully dynamic, no canned bracket sentences ----
        lines.append("**6. Recommendations**")
        lines.append("")
        if sorted_data:
            lines.extend(cls._generate_dynamic_recommendation(student_name, overall_pct, sorted_data))
        else:
            lines.append("- No recommendations available")

        formatted = "\n".join(lines)
        interpretation = re.sub(r'^\d+\.\s+', '', formatted, flags=re.MULTILINE)

        return {
            "formatted": formatted,
            "interpretation": interpretation,
            "sql": sql,
            "rows": len(result_df),
            "data": convert_to_serializable(result_df)
        }

    # ------------------------------------------------------------------
    # Distributions (gpa_distribution, gender_distribution, etc.)
    # ------------------------------------------------------------------
    @classmethod
    def _format_distribution(cls, query, df, query_type, entities, sql):
        lines = []
        label_col = df.columns[0]
        count_col = 'count' if 'count' in df.columns else (df.columns[1] if len(df.columns) > 1 else None)
        pct_col = 'percentage' if 'percentage' in df.columns else None

        total_records = int(df[count_col].sum()) if count_col and pd.api.types.is_numeric_dtype(df[count_col]) else len(df)

        lines.append("**1. Overview**")
        lines.append("")
        lines.append(f"This breakdown covers **{len(df)}** categories across **{total_records}** total records.")
        lines.append("")

        lines.append("**2. Breakdown**")
        lines.append("")
        sorted_df = df.sort_values(count_col, ascending=False) if count_col else df
        for _, row in sorted_df.iterrows():
            label = row.get(label_col, 'Unknown')
            count = row.get(count_col, '') if count_col else ''
            pct = f" ({row[pct_col]:.1f}%)" if pct_col and not pd.isna(row.get(pct_col)) else ""
            lines.append(f"- **{label}**: {count}{pct}")
        lines.append("")

        lines.append("**3. Insights**")
        lines.append("")
        if count_col and len(sorted_df) > 0:
            top = sorted_df.iloc[0]
            lines.append(f"- Most common: **{top[label_col]}** with **{top[count_col]}** records")
            if len(sorted_df) > 1:
                bottom = sorted_df.iloc[-1]
                lines.append(f"- Least common: **{bottom[label_col]}** with **{bottom[count_col]}** records")
            if len(sorted_df) > 1 and top[count_col] and bottom[count_col] is not None:
                try:
                    spread = top[count_col] - bottom[count_col]
                    lines.append(f"- Spread between most and least common: **{spread}** records")
                except Exception:
                    pass
        lines.append("")

        formatted = "\n".join(lines)
        interpretation = re.sub(r'^\d+\.\s+', '', formatted, flags=re.MULTILINE)
        return {
            "formatted": formatted,
            "interpretation": interpretation,
            "sql": sql,
            "rows": len(df),
            "data": convert_to_serializable(df)
        }

    # ------------------------------------------------------------------
    # Ranked lists (top_students, rank, performance_summary)
    # ------------------------------------------------------------------
    @classmethod
    def _format_ranked_list(cls, query, df, query_type, entities, sql):
        lines = []
        lines.append("**1. Overview**")
        lines.append("")
        lines.append(f"Showing **{len(df)}** ranked record(s).")
        lines.append("")

        lines.append("**2. Rankings**")
        lines.append("")
        for i, (_, row) in enumerate(df.iterrows(), start=1):
            name = row.get('name', f'Record {i}')
            pct = row.get('overall_percentage', row.get('percentage'))
            grade = row.get('overall_grade', row.get('grade'))
            gpa = row.get('overall_gpa', row.get('gpa'))
            parts = [f"#{row.get('rank', i)}", f"**{name}**"]
            if pct is not None and not pd.isna(pct):
                parts.append(f"{pct:.1f}%")
            if grade is not None and not pd.isna(grade):
                parts.append(f"Grade {grade}")
            if gpa is not None and not pd.isna(gpa):
                parts.append(f"GPA {gpa:.2f}")
            lines.append("- " + " - ".join(str(p) for p in parts))
        lines.append("")

        lines.append("**3. Summary**")
        lines.append("")
        if 'overall_percentage' in df.columns and len(df) > 0:
            spread = df['overall_percentage'].max() - df['overall_percentage'].min()
            lines.append(f"- Spread from top to bottom: **{spread:.1f} percentage points**")
            if len(df) > 1:
                gap_to_next = abs(df['overall_percentage'].iloc[0] - df['overall_percentage'].iloc[1])
                lines.append(f"- Gap between #1 and #2: **{gap_to_next:.1f} points**")
        lines.append("")

        formatted = "\n".join(lines)
        interpretation = re.sub(r'^\d+\.\s+', '', formatted, flags=re.MULTILINE)
        return {
            "formatted": formatted,
            "interpretation": interpretation,
            "sql": sql,
            "rows": len(df),
            "data": convert_to_serializable(df)
        }

    # ------------------------------------------------------------------
    # Single metric (gpa_calculation, total_marks, attendance)
    # ------------------------------------------------------------------
    @classmethod
    def _format_single_metric(cls, query, df, query_type, entities, sql):
        lines = []
        lines.append("**1. Result**")
        lines.append("")
        for _, row in df.iterrows():
            for col in df.columns:
                val = row[col]
                if pd.isna(val):
                    continue
                label = col.replace('_', ' ').title()
                if isinstance(val, float):
                    lines.append(f"- {label}: **{val:.2f}**")
                else:
                    lines.append(f"- {label}: **{val}**")
            lines.append("")

        formatted = "\n".join(lines)
        interpretation = re.sub(r'^\d+\.\s+', '', formatted, flags=re.MULTILINE)
        return {
            "formatted": formatted,
            "interpretation": interpretation,
            "sql": sql,
            "rows": len(df),
            "data": convert_to_serializable(df)
        }

    # ------------------------------------------------------------------
    # Fallback for lookup/list-shaped results (student_details, list_subjects, etc.)
    # ------------------------------------------------------------------
    @classmethod
    def _format_generic(cls, query: str, result_df: pd.DataFrame, query_type: str, entities: Dict, sql: str) -> Dict[str, Any]:
        try:
            headers = result_df.columns.tolist()
            table_data = []
            for _, row in result_df.iterrows():
                row_data = []
                for col in headers:
                    val = row[col]
                    if pd.isna(val) or val == "" or val is None:
                        val = "N/A"
                    elif isinstance(val, float):
                        val = int(val) if val == int(val) else round(val, 2)
                    row_data.append(val)
                table_data.append(row_data)

            formatted_table = tabulate(table_data, headers=headers, tablefmt="grid", stralign="left")
            lines = [f"**Found {len(result_df)} rows:**", "", formatted_table]
            if sql:
                lines += ["", "**Generated SQL:**", sql]

            return {
                "formatted": "\n".join(lines),
                "interpretation": f"Found **{len(result_df)}** record(s) matching your query.",
                "sql": sql,
                "rows": len(result_df),
                "data": convert_to_serializable(result_df)
            }
        except Exception as e:
            logger.error(f"Formatting error: {e}")
            return {
                "formatted": f"Found {len(result_df)} rows. Error formatting: {e}",
                "interpretation": f"Found {len(result_df)} record(s).",
                "sql": sql,
                "rows": len(result_df),
                "data": convert_to_serializable(result_df)
            }


# =============================================================================
# UNIVERSAL QUERY HANDLER
# =============================================================================

class UniversalQueryHandler:
    def __init__(self, dataframes: Dict[str, pd.DataFrame], rel_map: RelationshipMap):
        self.dataframes = dataframes
        self.rel_map = rel_map
        self.sql_gen = SQLGenerator(rel_map)
        self.sql_gen.create_tables_from_dataframes(dataframes)

    def handle_query(self, query: str) -> Dict[str, Any]:
        parsed = UniversalSemanticQueryParser.parse(query)
        query_type = parsed.get('query_type', 'general')
        entities = parsed.get('entities', {})

        if query_type == 'help':
            return {
                "success": True,
                "formatted": "Help message",
                "interpretation": "**1. Welcome to the Education Data Analysis System**\n\nI can help you analyze student data, marks, attendance, demographics, and more.",
                "rows": 0
            }

        if query_type == 'compare_students':
            requested = entities.get('students', [])
            if len(requested) >= 2:
                sql, resolved_names, not_found = MultiStudentSQLGenerator.generate(self.sql_gen, requested)
                if not_found:
                    return self._error_response(
                        f"Couldn't find student(s): {', '.join(not_found)}. Check spelling and try again."
                    )
                try:
                    df = self.sql_gen.execute_query(sql)
                    return ResultFormatter.format(
                        query=query,
                        result_df=df,
                        query_type=query_type,
                        entities=entities,
                        sql=sql,
                        students=resolved_names,
                    )
                except Exception as e:
                    logger.error(f"Multi-student query error: {e}")
                    return {
                        "success": False,
                        "error": str(e),
                        "sql": sql,
                        "formatted": f"Error executing comparison: {e}",
                        "interpretation": f"Failed to compare students: {e}",
                        "rows": 0
                    }

        handler_map = self._get_handler_map()
        handler = handler_map.get(query_type)
        if handler:
            return handler(parsed)

        # Raw-SQL fallback  gated by validate_sql
        try:
            df = self.sql_gen.execute_query(query, validate=True)
            return ResultFormatter.format(
                query=query,
                result_df=df,
                query_type='general',
                entities=entities,
                sql=query
            )
        except Exception as e:
            return self._general_response(query, str(e))

    def _get_handler_map(self):
        return {
            'student_details': self._handle_student_details,
            'student_list': self._handle_student_list,
            'student_count': self._handle_student_count,
            'student_by_gender': self._handle_student_by_gender,
            'student_by_age': self._handle_student_by_age,
            'student_contact': self._handle_student_contact,
            'student_family': self._handle_student_family,
            'student_address': self._handle_student_address,
            'student_marks': self._handle_student_marks,
            'marks_in_subject': self._handle_marks_in_subject,
            'highest_marks': self._handle_highest_marks,
            'lowest_marks': self._handle_lowest_marks,
            'average_marks': self._handle_average_marks,
            'total_marks': self._handle_total_marks,
            'marks_above_percentage': self._handle_marks_above_percentage,
            'marks_below_percentage': self._handle_marks_below_percentage,
            'gpa_calculation': self._handle_gpa_calculation,
            'gpa_distribution': self._handle_gpa_distribution,
            'gpa_filter': self._handle_gpa_filter,
            'grade_analysis': self._handle_grade_analysis,
            'pass_fail_analysis': self._handle_pass_fail_analysis,
            'failed_students': self._handle_failed_students,
            'passed_students': self._handle_passed_students,
            'attendance': self._handle_attendance,
            'attendance_above': self._handle_attendance_above,
            'attendance_below': self._handle_attendance_below,
            'top_students': self._handle_top_students,
            'bottom_students': self._handle_bottom_students,
            'rank': self._handle_rank,
            'list_subjects': self._handle_list_subjects,
            'subject_details': self._handle_subject_details,
            'subject_performance': self._handle_subject_performance,
            'list_exams': self._handle_list_exams,
            'exam_details': self._handle_exam_details,
            'section_students': self._handle_section_students,
            'section_strength': self._handle_section_strength,
            'ethnicity_distribution': self._handle_ethnicity_distribution,
            'religion_distribution': self._handle_religion_distribution,
            'blood_group_distribution': self._handle_blood_group_distribution,
            'gender_distribution': self._handle_gender_distribution,
            'age_distribution': self._handle_age_distribution,
            'address_distribution': self._handle_address_distribution,
            'statistics': self._handle_statistics,
            'passout_students': self._handle_passout_students,
            'academic_report': self._handle_academic_report,
            'performance_summary': self._handle_performance_summary,
            'complex_filter': self._handle_complex_filter,
        }

    # ------------------------------------------------------------------
    # HANDLERS
    # ------------------------------------------------------------------

    def _handle_student_marks(self, parsed: Dict) -> Dict:
        student = parsed['entities'].get('student_name')
        if not student:
            return self._error_response("Student name not found")

        sql = f"""
        SELECT s.name, sub.title AS subject,
               SUM(CAST(m.mark AS REAL)) AS obtained_mark,
               SUM(CAST(ep.fullMark AS REAL)) AS full_marks,
               ROUND(SUM(CAST(m.mark AS REAL)) * 100.0 / SUM(CAST(ep.fullMark AS REAL)), 2) AS percentage,
               GET_GRADE(SUM(CAST(m.mark AS REAL)) * 100.0 / SUM(CAST(ep.fullMark AS REAL))) AS grade
        FROM students s
        JOIN marks m ON s.uuid = m.student_id
        JOIN exam_parameters ep ON m.exam_parameter_id = ep.id
        JOIN subjects sub ON ep.class_section_subject_id = sub.cssId
        WHERE UPPER(s.name) LIKE UPPER('%{student}%')
        GROUP BY s.name, sub.title
        ORDER BY sub.title
        """
        try:
            df = self.sql_gen.execute_query(sql)
            return ResultFormatter.format(
                query=parsed['original_query'],
                result_df=df,
                query_type='student_marks',
                entities=parsed['entities'],
                sql=sql
            )
        except Exception as e:
            return self._error_response(f"Error fetching marks: {e}")

    def _handle_student_details(self, parsed: Dict) -> Dict:
        student = parsed['entities'].get('student_name')
        if not student:
            return self._error_response("Student name not found")
        sql = f"SELECT * FROM students WHERE UPPER(name) LIKE UPPER('%{student}%') ORDER BY name"
        try:
            df = self.sql_gen.execute_query(sql)
            return ResultFormatter.format(
                query=parsed['original_query'],
                result_df=df,
                query_type='student_details',
                entities=parsed['entities'],
                sql=sql
            )
        except Exception as e:
            return self._error_response(f"Error fetching student details: {e}")

    def _handle_student_list(self, parsed: Dict) -> Dict:
        sql = "SELECT name, gender, section_id, regd_no, mobile FROM students ORDER BY name"
        try:
            df = self.sql_gen.execute_query(sql)
            return ResultFormatter.format(
                query=parsed['original_query'],
                result_df=df,
                query_type='student_list',
                entities=parsed['entities'],
                sql=sql
            )
        except Exception as e:
            return self._error_response(f"Error fetching student list: {e}")

    def _handle_student_count(self, parsed: Dict) -> Dict:
        sql = "SELECT COUNT(*) AS total_students FROM students"
        try:
            df = self.sql_gen.execute_query(sql)
            return ResultFormatter.format(
                query=parsed['original_query'],
                result_df=df,
                query_type='student_count',
                entities=parsed['entities'],
                sql=sql
            )
        except Exception as e:
            return self._error_response(f"Error counting students: {e}")

    def _handle_student_by_gender(self, parsed: Dict) -> Dict:
        gender = parsed['entities'].get('gender', '')
        if not gender:
            return self._error_response("Gender not found")
        sql = f"SELECT name, gender, section_id FROM students WHERE UPPER(gender) = UPPER('{gender}') ORDER BY name"
        try:
            df = self.sql_gen.execute_query(sql)
            return ResultFormatter.format(
                query=parsed['original_query'],
                result_df=df,
                query_type='student_by_gender',
                entities=parsed['entities'],
                sql=sql
            )
        except Exception as e:
            return self._error_response(f"Error fetching students by gender: {e}")

    def _handle_student_by_age(self, parsed: Dict) -> Dict:
        age = parsed['entities'].get('age')
        if not age:
            return self._error_response("Age not found")
        sql = f"SELECT name, dob, CALCULATE_AGE(dob) AS age FROM students WHERE CALCULATE_AGE(dob) = {age} ORDER BY name"
        try:
            df = self.sql_gen.execute_query(sql)
            return ResultFormatter.format(
                query=parsed['original_query'],
                result_df=df,
                query_type='student_by_age',
                entities=parsed['entities'],
                sql=sql
            )
        except Exception as e:
            return self._error_response(f"Error fetching students by age: {e}")

    def _handle_student_contact(self, parsed: Dict) -> Dict:
        student = parsed['entities'].get('student_name')
        if student:
            sql = f"SELECT name, mobile, landline FROM students WHERE UPPER(name) LIKE UPPER('%{student}%') ORDER BY name"
        else:
            sql = "SELECT name, mobile, landline FROM students ORDER BY name"
        try:
            df = self.sql_gen.execute_query(sql)
            return ResultFormatter.format(
                query=parsed['original_query'],
                result_df=df,
                query_type='student_contact',
                entities=parsed['entities'],
                sql=sql
            )
        except Exception as e:
            return self._error_response(f"Error fetching contact details: {e}")

    def _handle_student_family(self, parsed: Dict) -> Dict:
        student = parsed['entities'].get('student_name')
        if student:
            sql = f"SELECT name, father, mother, guardian FROM students WHERE UPPER(name) LIKE UPPER('%{student}%') ORDER BY name"
        else:
            sql = "SELECT name, father, mother, guardian FROM students ORDER BY name"
        try:
            df = self.sql_gen.execute_query(sql)
            return ResultFormatter.format(
                query=parsed['original_query'],
                result_df=df,
                query_type='student_family',
                entities=parsed['entities'],
                sql=sql
            )
        except Exception as e:
            return self._error_response(f"Error fetching family details: {e}")

    def _handle_student_address(self, parsed: Dict) -> Dict:
        student = parsed['entities'].get('student_name')
        location = parsed['entities'].get('location')
        if student:
            sql = f"SELECT name, address, ward, district, province FROM students WHERE UPPER(name) LIKE UPPER('%{student}%')"
        elif location:
            sql = f"SELECT name, address, ward, district, province FROM students WHERE UPPER(address) LIKE UPPER('%{location}%') OR UPPER(ward) LIKE UPPER('%{location}%') OR UPPER(district) LIKE UPPER('%{location}%')"
        else:
            sql = "SELECT name, address, ward, district, province FROM students ORDER BY name"
        try:
            df = self.sql_gen.execute_query(sql)
            return ResultFormatter.format(
                query=parsed['original_query'],
                result_df=df,
                query_type='student_address',
                entities=parsed['entities'],
                sql=sql
            )
        except Exception as e:
            return self._error_response(f"Error fetching address details: {e}")

    def _handle_marks_in_subject(self, parsed: Dict) -> Dict:
        student = parsed['entities'].get('student_name')
        subject = parsed['entities'].get('subject')
        if not student or not subject:
            return self._error_response("Student or subject not found")
        subject_condition = SubjectFetcher.get_subject_condition(self.sql_gen, subject)
        sql = f"""
        SELECT s.name, sub.title AS subject,
               SUM(CAST(m.mark AS REAL)) AS obtained_mark,
               SUM(CAST(ep.fullMark AS REAL)) AS full_marks,
               ROUND(SUM(CAST(m.mark AS REAL)) * 100.0 / SUM(CAST(ep.fullMark AS REAL)), 2) AS percentage,
               GET_GRADE(SUM(CAST(m.mark AS REAL)) * 100.0 / SUM(CAST(ep.fullMark AS REAL))) AS grade
        FROM students s
        JOIN marks m ON s.uuid = m.student_id
        JOIN exam_parameters ep ON m.exam_parameter_id = ep.id
        JOIN subjects sub ON ep.class_section_subject_id = sub.cssId
        WHERE UPPER(s.name) LIKE UPPER('%{student}%') AND {subject_condition}
        GROUP BY s.name, sub.title
        """
        try:
            df = self.sql_gen.execute_query(sql)
            return ResultFormatter.format(
                query=parsed['original_query'],
                result_df=df,
                query_type='marks_in_subject',
                entities=parsed['entities'],
                sql=sql
            )
        except Exception as e:
            return self._error_response(f"Error fetching marks in subject: {e}")

    def _handle_highest_marks(self, parsed: Dict) -> Dict:
        subject = parsed['entities'].get('subject', 'all')
        if subject and subject.lower() != 'all':
            subject_condition = SubjectFetcher.get_subject_condition(self.sql_gen, subject)
            sql = f"""
            SELECT s.name, sub.title AS subject,
                   MAX(CAST(m.mark AS REAL)) AS highest_mark,
                   ep.fullMark AS full_marks,
                   GET_GRADE(MAX(CAST(m.mark AS REAL)) * 100.0 / CAST(ep.fullMark AS REAL)) AS grade
            FROM students s
            JOIN marks m ON s.uuid = m.student_id
            JOIN exam_parameters ep ON m.exam_parameter_id = ep.id
            JOIN subjects sub ON ep.class_section_subject_id = sub.cssId
            WHERE {subject_condition}
            GROUP BY s.name, sub.title, ep.fullMark
            ORDER BY highest_mark DESC
            LIMIT 1
            """
        else:
            sql = """
            SELECT s.name,
                   SUM(CAST(m.mark AS REAL)) AS total_marks,
                   SUM(CAST(ep.fullMark AS REAL)) AS full_marks,
                   ROUND(SUM(CAST(m.mark AS REAL)) * 100.0 / SUM(CAST(ep.fullMark AS REAL)), 2) AS percentage,
                   GET_GRADE(SUM(CAST(m.mark AS REAL)) * 100.0 / SUM(CAST(ep.fullMark AS REAL))) AS grade
            FROM students s
            JOIN marks m ON s.uuid = m.student_id
            JOIN exam_parameters ep ON m.exam_parameter_id = ep.id
            GROUP BY s.name
            ORDER BY total_marks DESC
            LIMIT 1
            """
        try:
            df = self.sql_gen.execute_query(sql)
            return ResultFormatter.format(
                query=parsed['original_query'],
                result_df=df,
                query_type='highest_marks',
                entities=parsed['entities'],
                sql=sql
            )
        except Exception as e:
            return self._error_response(f"Error fetching highest marks: {e}")

    def _handle_lowest_marks(self, parsed: Dict) -> Dict:
        subject = parsed['entities'].get('subject', 'all')
        if subject and subject.lower() != 'all':
            subject_condition = SubjectFetcher.get_subject_condition(self.sql_gen, subject)
            sql = f"""
            SELECT s.name, sub.title AS subject,
                   MIN(CAST(m.mark AS REAL)) AS lowest_mark,
                   ep.fullMark AS full_marks,
                   GET_GRADE(MIN(CAST(m.mark AS REAL)) * 100.0 / CAST(ep.fullMark AS REAL)) AS grade
            FROM students s
            JOIN marks m ON s.uuid = m.student_id
            JOIN exam_parameters ep ON m.exam_parameter_id = ep.id
            JOIN subjects sub ON ep.class_section_subject_id = sub.cssId
            WHERE {subject_condition}
            GROUP BY s.name, sub.title, ep.fullMark
            ORDER BY lowest_mark ASC
            LIMIT 1
            """
        else:
            sql = """
            SELECT s.name,
                   SUM(CAST(m.mark AS REAL)) AS total_marks,
                   SUM(CAST(ep.fullMark AS REAL)) AS full_marks,
                   ROUND(SUM(CAST(m.mark AS REAL)) * 100.0 / SUM(CAST(ep.fullMark AS REAL)), 2) AS percentage,
                   GET_GRADE(SUM(CAST(m.mark AS REAL)) * 100.0 / SUM(CAST(ep.fullMark AS REAL))) AS grade
            FROM students s
            JOIN marks m ON s.uuid = m.student_id
            JOIN exam_parameters ep ON m.exam_parameter_id = ep.id
            GROUP BY s.name
            ORDER BY total_marks ASC
            LIMIT 1
            """
        try:
            df = self.sql_gen.execute_query(sql)
            return ResultFormatter.format(
                query=parsed['original_query'],
                result_df=df,
                query_type='lowest_marks',
                entities=parsed['entities'],
                sql=sql
            )
        except Exception as e:
            return self._error_response(f"Error fetching lowest marks: {e}")

    def _handle_average_marks(self, parsed: Dict) -> Dict:
        sql = """
        SELECT sub.title AS subject,
               ROUND(AVG(CAST(m.mark AS REAL)), 2) AS average_marks,
               ROUND(AVG(CAST(m.mark AS REAL) * 100.0 / CAST(ep.fullMark AS REAL)), 2) AS average_percentage
        FROM marks m
        JOIN exam_parameters ep ON m.exam_parameter_id = ep.id
        JOIN subjects sub ON ep.class_section_subject_id = sub.cssId
        GROUP BY sub.title
        ORDER BY sub.title
        """
        try:
            df = self.sql_gen.execute_query(sql)
            return ResultFormatter.format(
                query=parsed['original_query'],
                result_df=df,
                query_type='average_marks',
                entities=parsed['entities'],
                sql=sql
            )
        except Exception as e:
            return self._error_response(f"Error fetching average marks: {e}")

    def _handle_total_marks(self, parsed: Dict) -> Dict:
        student = parsed['entities'].get('student_name')
        if student:
            sql = f"""
            SELECT s.name,
                   SUM(CAST(m.mark AS REAL)) AS total_marks,
                   SUM(CAST(ep.fullMark AS REAL)) AS full_marks,
                   ROUND(SUM(CAST(m.mark AS REAL)) * 100.0 / SUM(CAST(ep.fullMark AS REAL)), 2) AS percentage,
                   GET_GRADE(SUM(CAST(m.mark AS REAL)) * 100.0 / SUM(CAST(ep.fullMark AS REAL))) AS grade
            FROM students s
            JOIN marks m ON s.uuid = m.student_id
            JOIN exam_parameters ep ON m.exam_parameter_id = ep.id
            WHERE UPPER(s.name) LIKE UPPER('%{student}%')
            GROUP BY s.name
            """
        else:
            sql = """
            SELECT s.name,
                   SUM(CAST(m.mark AS REAL)) AS total_marks,
                   SUM(CAST(ep.fullMark AS REAL)) AS full_marks,
                   ROUND(SUM(CAST(m.mark AS REAL)) * 100.0 / SUM(CAST(ep.fullMark AS REAL)), 2) AS percentage,
                   GET_GRADE(SUM(CAST(m.mark AS REAL)) * 100.0 / SUM(CAST(ep.fullMark AS REAL))) AS grade
            FROM students s
            JOIN marks m ON s.uuid = m.student_id
            JOIN exam_parameters ep ON m.exam_parameter_id = ep.id
            GROUP BY s.name
            ORDER BY total_marks DESC
            """
        try:
            df = self.sql_gen.execute_query(sql)
            return ResultFormatter.format(
                query=parsed['original_query'],
                result_df=df,
                query_type='total_marks',
                entities=parsed['entities'],
                sql=sql
            )
        except Exception as e:
            return self._error_response(f"Error fetching total marks: {e}")

    def _handle_marks_above_percentage(self, parsed: Dict) -> Dict:
        percentage = parsed['entities'].get('percentage')
        subject = parsed['entities'].get('subject')
        if not percentage:
            return self._error_response("Percentage not found")
        if subject:
            subject_condition = SubjectFetcher.get_subject_condition(self.sql_gen, subject)
            sql = f"""
            SELECT s.name, sub.title AS subject,
                   ROUND(CAST(m.mark AS REAL) * 100.0 / CAST(ep.fullMark AS REAL), 2) AS percentage,
                   GET_GRADE(CAST(m.mark AS REAL) * 100.0 / CAST(ep.fullMark AS REAL)) AS grade
            FROM students s
            JOIN marks m ON s.uuid = m.student_id
            JOIN exam_parameters ep ON m.exam_parameter_id = ep.id
            JOIN subjects sub ON ep.class_section_subject_id = sub.cssId
            WHERE {subject_condition}
            AND CAST(m.mark AS REAL) * 100.0 / CAST(ep.fullMark AS REAL) > {percentage}
            ORDER BY percentage DESC
            """
        else:
            sql = f"""
            SELECT s.name,
                   ROUND(SUM(CAST(m.mark AS REAL)) * 100.0 / SUM(CAST(ep.fullMark AS REAL)), 2) AS overall_percentage,
                   GET_GRADE(SUM(CAST(m.mark AS REAL)) * 100.0 / SUM(CAST(ep.fullMark AS REAL))) AS grade
            FROM students s
            JOIN marks m ON s.uuid = m.student_id
            JOIN exam_parameters ep ON m.exam_parameter_id = ep.id
            GROUP BY s.name
            HAVING overall_percentage > {percentage}
            ORDER BY overall_percentage DESC
            """
        try:
            df = self.sql_gen.execute_query(sql)
            return ResultFormatter.format(
                query=parsed['original_query'],
                result_df=df,
                query_type='marks_above_percentage',
                entities=parsed['entities'],
                sql=sql
            )
        except Exception as e:
            return self._error_response(f"Error fetching marks above percentage: {e}")

    def _handle_marks_below_percentage(self, parsed: Dict) -> Dict:
        percentage = parsed['entities'].get('percentage')
        subject = parsed['entities'].get('subject')
        if not percentage:
            return self._error_response("Percentage not found")
        if subject:
            subject_condition = SubjectFetcher.get_subject_condition(self.sql_gen, subject)
            sql = f"""
            SELECT s.name, sub.title AS subject,
                   ROUND(CAST(m.mark AS REAL) * 100.0 / CAST(ep.fullMark AS REAL), 2) AS percentage,
                   GET_GRADE(CAST(m.mark AS REAL) * 100.0 / CAST(ep.fullMark AS REAL)) AS grade
            FROM students s
            JOIN marks m ON s.uuid = m.student_id
            JOIN exam_parameters ep ON m.exam_parameter_id = ep.id
            JOIN subjects sub ON ep.class_section_subject_id = sub.cssId
            WHERE {subject_condition}
            AND CAST(m.mark AS REAL) * 100.0 / CAST(ep.fullMark AS REAL) < {percentage}
            ORDER BY percentage ASC
            """
        else:
            sql = f"""
            SELECT s.name,
                   ROUND(SUM(CAST(m.mark AS REAL)) * 100.0 / SUM(CAST(ep.fullMark AS REAL)), 2) AS overall_percentage,
                   GET_GRADE(SUM(CAST(m.mark AS REAL)) * 100.0 / SUM(CAST(ep.fullMark AS REAL))) AS grade
            FROM students s
            JOIN marks m ON s.uuid = m.student_id
            JOIN exam_parameters ep ON m.exam_parameter_id = ep.id
            GROUP BY s.name
            HAVING overall_percentage < {percentage}
            ORDER BY overall_percentage ASC
            """
        try:
            df = self.sql_gen.execute_query(sql)
            return ResultFormatter.format(
                query=parsed['original_query'],
                result_df=df,
                query_type='marks_below_percentage',
                entities=parsed['entities'],
                sql=sql
            )
        except Exception as e:
            return self._error_response(f"Error fetching marks below percentage: {e}")

    def _handle_gpa_calculation(self, parsed: Dict) -> Dict:
        student = parsed['entities'].get('student_name')
        if student:
            sql = f"""
            SELECT s.name,
                   ROUND(SUM(CAST(m.mark AS REAL)) * 100.0 / SUM(CAST(ep.fullMark AS REAL)), 2) AS overall_percentage,
                   GET_GPA(SUM(CAST(m.mark AS REAL)) * 100.0 / SUM(CAST(ep.fullMark AS REAL))) AS overall_gpa,
                   GET_GRADE(SUM(CAST(m.mark AS REAL)) * 100.0 / SUM(CAST(ep.fullMark AS REAL))) AS overall_grade
            FROM students s
            JOIN marks m ON s.uuid = m.student_id
            JOIN exam_parameters ep ON m.exam_parameter_id = ep.id
            WHERE UPPER(s.name) LIKE UPPER('%{student}%')
            GROUP BY s.uuid, s.name
            """
        else:
            sql = """
            SELECT s.name,
                   ROUND(SUM(CAST(m.mark AS REAL)) * 100.0 / SUM(CAST(ep.fullMark AS REAL)), 2) AS overall_percentage,
                   GET_GPA(SUM(CAST(m.mark AS REAL)) * 100.0 / SUM(CAST(ep.fullMark AS REAL))) AS overall_gpa,
                   GET_GRADE(SUM(CAST(m.mark AS REAL)) * 100.0 / SUM(CAST(ep.fullMark AS REAL))) AS overall_grade
            FROM students s
            JOIN marks m ON s.uuid = m.student_id
            JOIN exam_parameters ep ON m.exam_parameter_id = ep.id
            GROUP BY s.uuid, s.name
            ORDER BY overall_gpa DESC
            """
        try:
            df = self.sql_gen.execute_query(sql)
            return ResultFormatter.format(
                query=parsed['original_query'],
                result_df=df,
                query_type='gpa_calculation',
                entities=parsed['entities'],
                sql=sql
            )
        except Exception as e:
            return self._error_response(f"Error calculating GPA: {e}")

    def _handle_gpa_distribution(self, parsed: Dict) -> Dict:
        sql = """
        WITH student_grades AS (
            SELECT s.uuid,
                   GET_GRADE(SUM(CAST(m.mark AS REAL)) * 100.0 / SUM(CAST(ep.fullMark AS REAL))) AS grade
            FROM students s
            JOIN marks m ON s.uuid = m.student_id
            JOIN exam_parameters ep ON m.exam_parameter_id = ep.id
            GROUP BY s.uuid
        )
        SELECT grade,
               COUNT(*) AS count,
               ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM student_grades), 2) AS percentage
        FROM student_grades
        GROUP BY grade
        ORDER BY grade
        """
        try:
            df = self.sql_gen.execute_query(sql)
            return ResultFormatter.format(
                query=parsed['original_query'],
                result_df=df,
                query_type='gpa_distribution',
                entities=parsed['entities'],
                sql=sql
            )
        except Exception as e:
            return self._error_response(f"Error fetching GPA distribution: {e}")

    def _handle_gpa_filter(self, parsed: Dict) -> Dict:
        entities = parsed.get('entities', {})
        if 'grade' in entities and entities['grade'] in GRADE_TO_GPA_MAP:
            threshold = GRADE_TO_GPA_MAP[entities['grade']]
            comparison = 'less' if any(w in str(parsed.get('matched_keywords', [])) for w in ['less', 'below', 'under']) else 'greater'
            if comparison == 'less':
                sql = f"""
                SELECT s.name, s.uuid,
                       ROUND(SUM(CAST(m.mark AS REAL)) * 100.0 / SUM(CAST(ep.fullMark AS REAL)), 2) AS overall_percentage,
                       GET_GPA(SUM(CAST(m.mark AS REAL)) * 100.0 / SUM(CAST(ep.fullMark AS REAL))) AS overall_gpa,
                       GET_GRADE(SUM(CAST(m.mark AS REAL)) * 100.0 / SUM(CAST(ep.fullMark AS REAL))) AS overall_grade
                FROM students s
                JOIN marks m ON s.uuid = m.student_id
                JOIN exam_parameters ep ON m.exam_parameter_id = ep.id
                GROUP BY s.uuid, s.name
                HAVING overall_gpa < {threshold}
                ORDER BY overall_gpa DESC
                """
            else:
                sql = f"""
                SELECT s.name, s.uuid,
                       ROUND(SUM(CAST(m.mark AS REAL)) * 100.0 / SUM(CAST(ep.fullMark AS REAL)), 2) AS overall_percentage,
                       GET_GPA(SUM(CAST(m.mark AS REAL)) * 100.0 / SUM(CAST(ep.fullMark AS REAL))) AS overall_gpa,
                       GET_GRADE(SUM(CAST(m.mark AS REAL)) * 100.0 / SUM(CAST(ep.fullMark AS REAL))) AS overall_grade
                FROM students s
                JOIN marks m ON s.uuid = m.student_id
                JOIN exam_parameters ep ON m.exam_parameter_id = ep.id
                GROUP BY s.uuid, s.name
                HAVING overall_gpa > {threshold}
                ORDER BY overall_gpa DESC
                """
            try:
                df = self.sql_gen.execute_query(sql)
                return ResultFormatter.format(
                    query=parsed['original_query'],
                    result_df=df,
                    query_type='gpa_filter',
                    entities=parsed['entities'],
                    sql=sql
                )
            except Exception as e:
                return self._error_response(f"Error filtering GPA: {e}")
        return self._error_response("No valid GPA filter found")

    def _handle_grade_analysis(self, parsed: Dict) -> Dict:
        grade = parsed['entities'].get('grade')
        if grade:
            sql = f"""
            SELECT s.name, s.uuid,
                   ROUND(SUM(CAST(m.mark AS REAL)) * 100.0 / SUM(CAST(ep.fullMark AS REAL)), 2) AS overall_percentage,
                   GET_GPA(SUM(CAST(m.mark AS REAL)) * 100.0 / SUM(CAST(ep.fullMark AS REAL))) AS overall_gpa,
                   GET_GRADE(SUM(CAST(m.mark AS REAL)) * 100.0 / SUM(CAST(ep.fullMark AS REAL))) AS overall_grade
            FROM students s
            JOIN marks m ON s.uuid = m.student_id
            JOIN exam_parameters ep ON m.exam_parameter_id = ep.id
            GROUP BY s.uuid, s.name
            HAVING overall_grade = '{grade}'
            ORDER BY s.name
            """
        else:
            sql = """
            SELECT GET_GRADE(SUM(CAST(m.mark AS REAL)) * 100.0 / SUM(CAST(ep.fullMark AS REAL))) AS grade,
                   COUNT(*) AS count,
                   ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM students), 2) AS percentage
            FROM students s
            JOIN marks m ON s.uuid = m.student_id
            JOIN exam_parameters ep ON m.exam_parameter_id = ep.id
            GROUP BY grade
            ORDER BY grade
            """
        try:
            df = self.sql_gen.execute_query(sql)
            return ResultFormatter.format(
                query=parsed['original_query'],
                result_df=df,
                query_type='grade_analysis',
                entities=parsed['entities'],
                sql=sql
            )
        except Exception as e:
            return self._error_response(f"Error analyzing grades: {e}")

    def _handle_pass_fail_analysis(self, parsed: Dict) -> Dict:
        subject = parsed['entities'].get('subject')
        if subject:
            subject_condition = SubjectFetcher.get_subject_condition(self.sql_gen, subject)
            sql = f"""
            SELECT sub.title AS subject,
                   ROUND(COUNT(CASE WHEN CAST(m.mark AS REAL) >= CAST(ep.passMark AS REAL) THEN 1 END) * 100.0 / COUNT(*), 2) AS pass_percentage,
                   ROUND(COUNT(CASE WHEN CAST(m.mark AS REAL) < CAST(ep.passMark AS REAL) THEN 1 END) * 100.0 / COUNT(*), 2) AS fail_percentage,
                   COUNT(*) AS total_students,
                   SUM(CASE WHEN CAST(m.mark AS REAL) >= CAST(ep.passMark AS REAL) THEN 1 ELSE 0 END) AS passed,
                   SUM(CASE WHEN CAST(m.mark AS REAL) < CAST(ep.passMark AS REAL) THEN 1 ELSE 0 END) AS failed
            FROM marks m
            JOIN exam_parameters ep ON m.exam_parameter_id = ep.id
            JOIN subjects sub ON ep.class_section_subject_id = sub.cssId
            WHERE {subject_condition}
            GROUP BY sub.title
            """
        else:
            sql = """
            SELECT sub.title AS subject,
                   ROUND(COUNT(CASE WHEN CAST(m.mark AS REAL) >= CAST(ep.passMark AS REAL) THEN 1 END) * 100.0 / COUNT(*), 2) AS pass_percentage,
                   ROUND(COUNT(CASE WHEN CAST(m.mark AS REAL) < CAST(ep.passMark AS REAL) THEN 1 END) * 100.0 / COUNT(*), 2) AS fail_percentage,
                   COUNT(*) AS total_students,
                   SUM(CASE WHEN CAST(m.mark AS REAL) >= CAST(ep.passMark AS REAL) THEN 1 ELSE 0 END) AS passed,
                   SUM(CASE WHEN CAST(m.mark AS REAL) < CAST(ep.passMark AS REAL) THEN 1 ELSE 0 END) AS failed
            FROM marks m
            JOIN exam_parameters ep ON m.exam_parameter_id = ep.id
            JOIN subjects sub ON ep.class_section_subject_id = sub.cssId
            GROUP BY sub.title
            ORDER BY sub.title
            """
        try:
            df = self.sql_gen.execute_query(sql)
            return ResultFormatter.format(
                query=parsed['original_query'],
                result_df=df,
                query_type='pass_fail_analysis',
                entities=parsed['entities'],
                sql=sql
            )
        except Exception as e:
            return self._error_response(f"Error analyzing pass/fail: {e}")

    def _handle_failed_students(self, parsed: Dict) -> Dict:
        subject = parsed['entities'].get('subject')
        if subject and subject != 'any':
            subject_condition = SubjectFetcher.get_subject_condition(self.sql_gen, subject)
            sql = f"""
            SELECT s.name, s.uuid, sub.title AS subject,
                   CAST(m.mark AS REAL) AS marks,
                   CAST(ep.passMark AS REAL) AS pass_mark,
                   GET_GRADE(CAST(m.mark AS REAL) * 100.0 / CAST(ep.fullMark AS REAL)) AS grade
            FROM students s
            JOIN marks m ON s.uuid = m.student_id
            JOIN exam_parameters ep ON m.exam_parameter_id = ep.id
            JOIN subjects sub ON ep.class_section_subject_id = sub.cssId
            WHERE {subject_condition} AND CAST(m.mark AS REAL) < CAST(ep.passMark AS REAL)
            ORDER BY s.name, sub.title
            """
        else:
            sql = """
            SELECT s.name, s.uuid,
                   COUNT(*) AS failed_subjects,
                   GROUP_CONCAT(sub.title) AS subjects_failed
            FROM students s
            JOIN marks m ON s.uuid = m.student_id
            JOIN exam_parameters ep ON m.exam_parameter_id = ep.id
            JOIN subjects sub ON ep.class_section_subject_id = sub.cssId
            WHERE CAST(m.mark AS REAL) < CAST(ep.passMark AS REAL)
            GROUP BY s.uuid, s.name
            ORDER BY failed_subjects DESC
            """
        try:
            df = self.sql_gen.execute_query(sql)
            return ResultFormatter.format(
                query=parsed['original_query'],
                result_df=df,
                query_type='failed_students',
                entities=parsed['entities'],
                sql=sql
            )
        except Exception as e:
            return self._error_response(f"Error fetching failed students: {e}")

    def _handle_passed_students(self, parsed: Dict) -> Dict:
        subject = parsed['entities'].get('subject')
        if subject and subject != 'any':
            subject_condition = SubjectFetcher.get_subject_condition(self.sql_gen, subject)
            sql = f"""
            SELECT s.name, s.uuid, sub.title AS subject,
                   CAST(m.mark AS REAL) AS marks,
                   CAST(ep.passMark AS REAL) AS pass_mark,
                   GET_GRADE(CAST(m.mark AS REAL) * 100.0 / CAST(ep.fullMark AS REAL)) AS grade
            FROM students s
            JOIN marks m ON s.uuid = m.student_id
            JOIN exam_parameters ep ON m.exam_parameter_id = ep.id
            JOIN subjects sub ON ep.class_section_subject_id = sub.cssId
            WHERE {subject_condition} AND CAST(m.mark AS REAL) >= CAST(ep.passMark AS REAL)
            ORDER BY s.name, sub.title
            """
        else:
            sql = """
            SELECT DISTINCT s.name, s.uuid
            FROM students s
            JOIN marks m ON s.uuid = m.student_id
            JOIN exam_parameters ep ON m.exam_parameter_id = ep.id
            WHERE CAST(m.mark AS REAL) >= CAST(ep.passMark AS REAL)
            ORDER BY s.name
            """
        try:
            df = self.sql_gen.execute_query(sql)
            return ResultFormatter.format(
                query=parsed['original_query'],
                result_df=df,
                query_type='passed_students',
                entities=parsed['entities'],
                sql=sql
            )
        except Exception as e:
            return self._error_response(f"Error fetching passed students: {e}")

    def _handle_attendance(self, parsed: Dict) -> Dict:
        student = parsed['entities'].get('student_name')
        if student:
            sql = f"""
            SELECT s.name, a.attendance,
                   ROUND(CAST(a.attendance AS REAL) * 100.0 / 59, 2) AS attendance_percentage
            FROM students s
            JOIN attendances a ON s.id = a.student_id
            WHERE UPPER(s.name) LIKE UPPER('%{student}%')
            """
        else:
            sql = """
            SELECT s.name, a.attendance,
                   ROUND(CAST(a.attendance AS REAL) * 100.0 / 59, 2) AS attendance_percentage
            FROM students s
            JOIN attendances a ON s.id = a.student_id
            ORDER BY attendance_percentage DESC
            """
        try:
            df = self.sql_gen.execute_query(sql)
            return ResultFormatter.format(
                query=parsed['original_query'],
                result_df=df,
                query_type='attendance',
                entities=parsed['entities'],
                sql=sql
            )
        except Exception as e:
            return self._error_response(f"Error fetching attendance: {e}")

    def _handle_attendance_above(self, parsed: Dict) -> Dict:
        threshold = parsed['entities'].get('attendance_threshold', 80)
        sql = f"""
        SELECT s.name, a.attendance,
               ROUND(CAST(a.attendance AS REAL) * 100.0 / 59, 2) AS attendance_percentage
        FROM students s
        JOIN attendances a ON s.id = a.student_id
        WHERE CAST(a.attendance AS REAL) > {threshold}
        ORDER BY attendance_percentage DESC
        """
        try:
            df = self.sql_gen.execute_query(sql)
            return ResultFormatter.format(
                query=parsed['original_query'],
                result_df=df,
                query_type='attendance_above',
                entities=parsed['entities'],
                sql=sql
            )
        except Exception as e:
            return self._error_response(f"Error fetching attendance above: {e}")

    def _handle_attendance_below(self, parsed: Dict) -> Dict:
        threshold = parsed['entities'].get('attendance_threshold', 80)
        sql = f"""
        SELECT s.name, a.attendance,
               ROUND(CAST(a.attendance AS REAL) * 100.0 / 59, 2) AS attendance_percentage
        FROM students s
        JOIN attendances a ON s.id = a.student_id
        WHERE CAST(a.attendance AS REAL) < {threshold}
        ORDER BY attendance_percentage ASC
        """
        try:
            df = self.sql_gen.execute_query(sql)
            return ResultFormatter.format(
                query=parsed['original_query'],
                result_df=df,
                query_type='attendance_below',
                entities=parsed['entities'],
                sql=sql
            )
        except Exception as e:
            return self._error_response(f"Error fetching attendance below: {e}")

    def _handle_top_students(self, parsed: Dict) -> Dict:
        limit = parsed['entities'].get('limit', 10)
        sql = f"""
        SELECT s.name, s.uuid,
               ROUND(SUM(CAST(m.mark AS REAL)) * 100.0 / SUM(CAST(ep.fullMark AS REAL)), 2) AS overall_percentage,
               GET_GPA(SUM(CAST(m.mark AS REAL)) * 100.0 / SUM(CAST(ep.fullMark AS REAL))) AS overall_gpa,
               GET_GRADE(SUM(CAST(m.mark AS REAL)) * 100.0 / SUM(CAST(ep.fullMark AS REAL))) AS overall_grade
        FROM students s
        JOIN marks m ON s.uuid = m.student_id
        JOIN exam_parameters ep ON m.exam_parameter_id = ep.id
        GROUP BY s.uuid, s.name
        ORDER BY overall_percentage DESC
        LIMIT {limit}
        """
        try:
            df = self.sql_gen.execute_query(sql)
            return ResultFormatter.format(
                query=parsed['original_query'],
                result_df=df,
                query_type='top_students',
                entities=parsed['entities'],
                sql=sql
            )
        except Exception as e:
            return self._error_response(f"Error fetching top students: {e}")

    def _handle_bottom_students(self, parsed: Dict) -> Dict:
        limit = parsed['entities'].get('limit', 10)
        sql = f"""
        SELECT s.name, s.uuid,
               ROUND(SUM(CAST(m.mark AS REAL)) * 100.0 / SUM(CAST(ep.fullMark AS REAL)), 2) AS overall_percentage,
               GET_GPA(SUM(CAST(m.mark AS REAL)) * 100.0 / SUM(CAST(ep.fullMark AS REAL))) AS overall_gpa,
               GET_GRADE(SUM(CAST(m.mark AS REAL)) * 100.0 / SUM(CAST(ep.fullMark AS REAL))) AS overall_grade
        FROM students s
        JOIN marks m ON s.uuid = m.student_id
        JOIN exam_parameters ep ON m.exam_parameter_id = ep.id
        GROUP BY s.uuid, s.name
        ORDER BY overall_percentage ASC
        LIMIT {limit}
        """
        try:
            df = self.sql_gen.execute_query(sql)
            return ResultFormatter.format(
                query=parsed['original_query'],
                result_df=df,
                query_type='bottom_students',
                entities=parsed['entities'],
                sql=sql
            )
        except Exception as e:
            return self._error_response(f"Error fetching bottom students: {e}")

    def _handle_rank(self, parsed: Dict) -> Dict:
        student = parsed['entities'].get('student_name')
        if student:
            sql = f"""
            WITH ranked_students AS (
                SELECT
                    s.name,
                    s.uuid,
                    ROUND(SUM(CAST(m.mark AS REAL)) * 100.0 / SUM(CAST(ep.fullMark AS REAL)), 2) AS percentage,
                    GET_GPA(SUM(CAST(m.mark AS REAL)) * 100.0 / SUM(CAST(ep.fullMark AS REAL))) AS gpa,
                    GET_GRADE(SUM(CAST(m.mark AS REAL)) * 100.0 / SUM(CAST(ep.fullMark AS REAL))) AS grade,
                    ROW_NUMBER() OVER (ORDER BY SUM(CAST(m.mark AS REAL)) * 100.0 / SUM(CAST(ep.fullMark AS REAL)) DESC) AS rank,
                    COUNT(*) OVER () AS total_students
                FROM students s
                JOIN marks m ON s.uuid = m.student_id
                JOIN exam_parameters ep ON m.exam_parameter_id = ep.id
                GROUP BY s.name, s.uuid
            )
            SELECT * FROM ranked_students
            WHERE UPPER(name) LIKE UPPER('%{student}%')
            """
        else:
            sql = """
            WITH ranked_students AS (
                SELECT
                    s.name,
                    s.uuid,
                    ROUND(SUM(CAST(m.mark AS REAL)) * 100.0 / SUM(CAST(ep.fullMark AS REAL)), 2) AS percentage,
                    GET_GPA(SUM(CAST(m.mark AS REAL)) * 100.0 / SUM(CAST(ep.fullMark AS REAL))) AS gpa,
                    GET_GRADE(SUM(CAST(m.mark AS REAL)) * 100.0 / SUM(CAST(ep.fullMark AS REAL))) AS grade,
                    ROW_NUMBER() OVER (ORDER BY SUM(CAST(m.mark AS REAL)) * 100.0 / SUM(CAST(ep.fullMark AS REAL)) DESC) AS rank
                FROM students s
                JOIN marks m ON s.uuid = m.student_id
                JOIN exam_parameters ep ON m.exam_parameter_id = ep.id
                GROUP BY s.name, s.uuid
            )
            SELECT * FROM ranked_students
            ORDER BY rank
            """
        try:
            df = self.sql_gen.execute_query(sql)
            return ResultFormatter.format(
                query=parsed['original_query'],
                result_df=df,
                query_type='rank',
                entities=parsed['entities'],
                sql=sql
            )
        except Exception as e:
            return self._error_response(f"Error fetching rank: {e}")

    def _handle_list_subjects(self, parsed: Dict) -> Dict:
        sql = "SELECT title, code, th_credit_hr, in_credit_hr FROM subjects ORDER BY title"
        try:
            df = self.sql_gen.execute_query(sql)
            return ResultFormatter.format(
                query=parsed['original_query'],
                result_df=df,
                query_type='list_subjects',
                entities=parsed['entities'],
                sql=sql
            )
        except Exception as e:
            return self._error_response(f"Error fetching subjects: {e}")

    def _handle_subject_details(self, parsed: Dict) -> Dict:
        subject = parsed['entities'].get('subject')
        if subject:
            subject_condition = SubjectFetcher.get_subject_condition(self.sql_gen, subject)
            sql = f"SELECT * FROM subjects WHERE {subject_condition}"
        else:
            sql = "SELECT * FROM subjects ORDER BY title"
        try:
            df = self.sql_gen.execute_query(sql)
            return ResultFormatter.format(
                query=parsed['original_query'],
                result_df=df,
                query_type='subject_details',
                entities=parsed['entities'],
                sql=sql
            )
        except Exception as e:
            return self._error_response(f"Error fetching subject details: {e}")

    def _handle_subject_performance(self, parsed: Dict) -> Dict:
        subject = parsed['entities'].get('subject')
        if subject:
            subject_condition = SubjectFetcher.get_subject_condition(self.sql_gen, subject)
            sql = f"""
            SELECT s.name, sub.title AS subject,
                   SUM(CAST(m.mark AS REAL)) AS obtained_mark,
                   SUM(CAST(ep.fullMark AS REAL)) AS full_marks,
                   ROUND(SUM(CAST(m.mark AS REAL)) * 100.0 / SUM(CAST(ep.fullMark AS REAL)), 2) AS percentage,
                   GET_GRADE(SUM(CAST(m.mark AS REAL)) * 100.0 / SUM(CAST(ep.fullMark AS REAL))) AS grade
            FROM students s
            JOIN marks m ON s.uuid = m.student_id
            JOIN exam_parameters ep ON m.exam_parameter_id = ep.id
            JOIN subjects sub ON ep.class_section_subject_id = sub.cssId
            WHERE {subject_condition}
            GROUP BY s.name, sub.title
            ORDER BY percentage DESC
            """
        else:
            sql = """
            SELECT sub.title AS subject,
                   ROUND(AVG(SUM(CAST(m.mark AS REAL))), 2) AS average_marks,
                   ROUND(AVG(SUM(CAST(m.mark AS REAL)) * 100.0 / SUM(CAST(ep.fullMark AS REAL))), 2) AS average_percentage,
                   COUNT(DISTINCT s.uuid) AS student_count
            FROM students s
            JOIN marks m ON s.uuid = m.student_id
            JOIN exam_parameters ep ON m.exam_parameter_id = ep.id
            JOIN subjects sub ON ep.class_section_subject_id = sub.cssId
            GROUP BY sub.title
            ORDER BY average_percentage DESC
            """
        try:
            df = self.sql_gen.execute_query(sql)
            return ResultFormatter.format(
                query=parsed['original_query'],
                result_df=df,
                query_type='subject_performance',
                entities=parsed['entities'],
                sql=sql
            )
        except Exception as e:
            return self._error_response(f"Error fetching subject performance: {e}")

    def _handle_list_exams(self, parsed: Dict) -> Dict:
        sql = "SELECT id, title, type_title, weightage, from_date, to_date FROM exams ORDER BY id"
        try:
            df = self.sql_gen.execute_query(sql)
            return ResultFormatter.format(
                query=parsed['original_query'],
                result_df=df,
                query_type='list_exams',
                entities=parsed['entities'],
                sql=sql
            )
        except Exception as e:
            return self._error_response(f"Error fetching exams: {e}")

    def _handle_exam_details(self, parsed: Dict) -> Dict:
        exam = parsed['entities'].get('exam_name')
        if exam:
            sql = f"SELECT * FROM exams WHERE UPPER(title) LIKE UPPER('%{exam}%')"
        else:
            sql = "SELECT * FROM exams ORDER BY id"
        try:
            df = self.sql_gen.execute_query(sql)
            return ResultFormatter.format(
                query=parsed['original_query'],
                result_df=df,
                query_type='exam_details',
                entities=parsed['entities'],
                sql=sql
            )
        except Exception as e:
            return self._error_response(f"Error fetching exam details: {e}")

    def _handle_section_students(self, parsed: Dict) -> Dict:
        section = parsed['entities'].get('section')
        if section:
            sql = f"""
            SELECT s.* FROM students s
            JOIN sections sec ON s.section_id = sec.id
            WHERE UPPER(sec.title) LIKE UPPER('%{section}%')
            ORDER BY s.name
            """
        else:
            sql = """
            SELECT sec.title AS section, COUNT(s.id) AS student_count
            FROM sections sec
            LEFT JOIN students s ON sec.id = s.section_id
            GROUP BY sec.id, sec.title
            ORDER BY student_count DESC
            """
        try:
            df = self.sql_gen.execute_query(sql)
            return ResultFormatter.format(
                query=parsed['original_query'],
                result_df=df,
                query_type='section_students',
                entities=parsed['entities'],
                sql=sql
            )
        except Exception as e:
            return self._error_response(f"Error fetching section students: {e}")

    def _handle_section_strength(self, parsed: Dict) -> Dict:
        sql = """
        SELECT sec.title AS section, COUNT(s.id) AS student_count
        FROM sections sec
        LEFT JOIN students s ON sec.id = s.section_id
        GROUP BY sec.id, sec.title
        ORDER BY student_count DESC
        """
        try:
            df = self.sql_gen.execute_query(sql)
            return ResultFormatter.format(
                query=parsed['original_query'],
                result_df=df,
                query_type='section_strength',
                entities=parsed['entities'],
                sql=sql
            )
        except Exception as e:
            return self._error_response(f"Error fetching section strength: {e}")

    def _handle_ethnicity_distribution(self, parsed: Dict) -> Dict:
        sql = """
        SELECT ethnicity,
               COUNT(*) AS count,
               ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM students), 2) AS percentage
        FROM students
        WHERE ethnicity != '' AND ethnicity IS NOT NULL
        GROUP BY ethnicity
        ORDER BY count DESC
        """
        try:
            df = self.sql_gen.execute_query(sql)
            return ResultFormatter.format(
                query=parsed['original_query'],
                result_df=df,
                query_type='ethnicity_distribution',
                entities=parsed['entities'],
                sql=sql
            )
        except Exception as e:
            return self._error_response(f"Error fetching ethnicity distribution: {e}")

    def _handle_religion_distribution(self, parsed: Dict) -> Dict:
        sql = """
        SELECT religion,
               COUNT(*) AS count,
               ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM students), 2) AS percentage
        FROM students
        WHERE religion != '' AND religion IS NOT NULL
        GROUP BY religion
        ORDER BY count DESC
        """
        try:
            df = self.sql_gen.execute_query(sql)
            return ResultFormatter.format(
                query=parsed['original_query'],
                result_df=df,
                query_type='religion_distribution',
                entities=parsed['entities'],
                sql=sql
            )
        except Exception as e:
            return self._error_response(f"Error fetching religion distribution: {e}")

    def _handle_blood_group_distribution(self, parsed: Dict) -> Dict:
        sql = """
        SELECT blood_group,
               COUNT(*) AS count,
               ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM students), 2) AS percentage
        FROM students
        WHERE blood_group != '' AND blood_group IS NOT NULL
        GROUP BY blood_group
        ORDER BY count DESC
        """
        try:
            df = self.sql_gen.execute_query(sql)
            return ResultFormatter.format(
                query=parsed['original_query'],
                result_df=df,
                query_type='blood_group_distribution',
                entities=parsed['entities'],
                sql=sql
            )
        except Exception as e:
            return self._error_response(f"Error fetching blood group distribution: {e}")

    def _handle_gender_distribution(self, parsed: Dict) -> Dict:
        sql = """
        SELECT gender,
               COUNT(*) AS count,
               ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM students), 2) AS percentage
        FROM students
        WHERE gender != '' AND gender IS NOT NULL
        GROUP BY gender
        ORDER BY count DESC
        """
        try:
            df = self.sql_gen.execute_query(sql)
            return ResultFormatter.format(
                query=parsed['original_query'],
                result_df=df,
                query_type='gender_distribution',
                entities=parsed['entities'],
                sql=sql
            )
        except Exception as e:
            return self._error_response(f"Error fetching gender distribution: {e}")

    def _handle_age_distribution(self, parsed: Dict) -> Dict:
        sql = """
        SELECT CALCULATE_AGE(dob) AS age,
               COUNT(*) AS count
        FROM students
        WHERE dob IS NOT NULL
        GROUP BY age
        ORDER BY age
        """
        try:
            df = self.sql_gen.execute_query(sql)
            return ResultFormatter.format(
                query=parsed['original_query'],
                result_df=df,
                query_type='age_distribution',
                entities=parsed['entities'],
                sql=sql
            )
        except Exception as e:
            return self._error_response(f"Error fetching age distribution: {e}")

    def _handle_address_distribution(self, parsed: Dict) -> Dict:
        query = parsed.get('original_query', '').lower()
        if 'district' in query:
            sql = """
            SELECT district,
                   COUNT(*) AS count,
                   ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM students), 2) AS percentage
            FROM students
            WHERE district != '' AND district IS NOT NULL
            GROUP BY district
            ORDER BY count DESC
            """
        elif 'province' in query:
            sql = """
            SELECT province,
                   COUNT(*) AS count,
                   ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM students), 2) AS percentage
            FROM students
            WHERE province != '' AND province IS NOT NULL
            GROUP BY province
            ORDER BY count DESC
            """
        elif 'ward' in query:
            sql = """
            SELECT ward,
                   COUNT(*) AS count,
                   ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM students), 2) AS percentage
            FROM students
            WHERE ward != '' AND ward IS NOT NULL
            GROUP BY ward
            ORDER BY count DESC
            """
        else:
            sql = """
            SELECT address,
                   COUNT(*) AS count
            FROM students
            WHERE address != '' AND address IS NOT NULL
            GROUP BY address
            ORDER BY count DESC
            LIMIT 20
            """
        try:
            df = self.sql_gen.execute_query(sql)
            return ResultFormatter.format(
                query=parsed['original_query'],
                result_df=df,
                query_type='address_distribution',
                entities=parsed['entities'],
                sql=sql
            )
        except Exception as e:
            return self._error_response(f"Error fetching address distribution: {e}")

    def _handle_statistics(self, parsed: Dict) -> Dict:
        sql = """
        SELECT
            (SELECT COUNT(*) FROM students) AS total_students,
            (SELECT COUNT(*) FROM students WHERE UPPER(gender) = 'MALE') AS male_students,
            (SELECT COUNT(*) FROM students WHERE UPPER(gender) = 'FEMALE') AS female_students,
            (SELECT COUNT(*) FROM subjects) AS total_subjects,
            (SELECT COUNT(*) FROM exams) AS total_exams,
            (SELECT COUNT(*) FROM marks) AS total_marks,
            (SELECT ROUND(AVG(CAST(m.mark AS REAL)), 2) FROM marks m) AS avg_marks,
            (SELECT ROUND(AVG(CAST(a.attendance AS REAL)), 2) FROM attendances a) AS avg_attendance,
            (SELECT COUNT(DISTINCT ethnicity) FROM students WHERE ethnicity != '' AND ethnicity IS NOT NULL) AS ethnicities,
            (SELECT COUNT(DISTINCT religion) FROM students WHERE religion != '' AND religion IS NOT NULL) AS religions,
            (SELECT COUNT(DISTINCT blood_group) FROM students WHERE blood_group != '' AND blood_group IS NOT NULL) AS blood_groups
        """
        try:
            df = self.sql_gen.execute_query(sql)
            return ResultFormatter.format(
                query=parsed['original_query'],
                result_df=df,
                query_type='statistics',
                entities=parsed['entities'],
                sql=sql
            )
        except Exception as e:
            return self._error_response(f"Error fetching statistics: {e}")

    def _handle_passout_students(self, parsed: Dict) -> Dict:
        sql = "SELECT * FROM students WHERE class_id = 60 ORDER BY name"
        try:
            df = self.sql_gen.execute_query(sql)
            return ResultFormatter.format(
                query=parsed['original_query'],
                result_df=df,
                query_type='passout_students',
                entities=parsed['entities'],
                sql=sql
            )
        except Exception as e:
            return self._error_response(f"Error fetching passout students: {e}")

    def _handle_academic_report(self, parsed: Dict) -> Dict:
        student = parsed['entities'].get('student_name')
        if student:
            sql = f"""
            SELECT s.name, sub.title AS subject,
                   SUM(CAST(m.mark AS REAL)) AS obtained_mark,
                   SUM(CAST(ep.fullMark AS REAL)) AS full_marks,
                   ROUND(SUM(CAST(m.mark AS REAL)) * 100.0 / SUM(CAST(ep.fullMark AS REAL)), 2) AS percentage,
                   GET_GRADE(SUM(CAST(m.mark AS REAL)) * 100.0 / SUM(CAST(ep.fullMark AS REAL))) AS grade,
                   GET_GPA(SUM(CAST(m.mark AS REAL)) * 100.0 / SUM(CAST(ep.fullMark AS REAL))) AS gpa
            FROM students s
            JOIN marks m ON s.uuid = m.student_id
            JOIN exam_parameters ep ON m.exam_parameter_id = ep.id
            JOIN subjects sub ON ep.class_section_subject_id = sub.cssId
            WHERE UPPER(s.name) LIKE UPPER('%{student}%')
            GROUP BY s.name, sub.title
            ORDER BY sub.title
            """
        else:
            sql = """
            SELECT s.name, sub.title AS subject,
                   SUM(CAST(m.mark AS REAL)) AS obtained_mark,
                   SUM(CAST(ep.fullMark AS REAL)) AS full_marks,
                   ROUND(SUM(CAST(m.mark AS REAL)) * 100.0 / SUM(CAST(ep.fullMark AS REAL)), 2) AS percentage,
                   GET_GRADE(SUM(CAST(m.mark AS REAL)) * 100.0 / SUM(CAST(ep.fullMark AS REAL))) AS grade
            FROM students s
            JOIN marks m ON s.uuid = m.student_id
            JOIN exam_parameters ep ON m.exam_parameter_id = ep.id
            JOIN subjects sub ON ep.class_section_subject_id = sub.cssId
            GROUP BY s.name, sub.title
            ORDER BY s.name, sub.title
            """
        try:
            df = self.sql_gen.execute_query(sql)
            return ResultFormatter.format(
                query=parsed['original_query'],
                result_df=df,
                query_type='academic_report',
                entities=parsed['entities'],
                sql=sql
            )
        except Exception as e:
            return self._error_response(f"Error fetching academic report: {e}")

    def _handle_performance_summary(self, parsed: Dict) -> Dict:
        sql = """
        SELECT s.name,
               ROUND(SUM(CAST(m.mark AS REAL)) * 100.0 / SUM(CAST(ep.fullMark AS REAL)), 2) AS overall_percentage,
               GET_GPA(SUM(CAST(m.mark AS REAL)) * 100.0 / SUM(CAST(ep.fullMark AS REAL))) AS overall_gpa,
               GET_GRADE(SUM(CAST(m.mark AS REAL)) * 100.0 / SUM(CAST(ep.fullMark AS REAL))) AS overall_grade,
               COUNT(DISTINCT sub.title) AS subjects_taken
        FROM students s
        JOIN marks m ON s.uuid = m.student_id
        JOIN exam_parameters ep ON m.exam_parameter_id = ep.id
        JOIN subjects sub ON ep.class_section_subject_id = sub.cssId
        GROUP BY s.uuid, s.name
        ORDER BY overall_percentage DESC
        """
        try:
            df = self.sql_gen.execute_query(sql)
            return ResultFormatter.format(
                query=parsed['original_query'],
                result_df=df,
                query_type='performance_summary',
                entities=parsed['entities'],
                sql=sql
            )
        except Exception as e:
            return self._error_response(f"Error fetching performance summary: {e}")

    def _handle_complex_filter(self, parsed: Dict) -> Dict:
        query = parsed.get('original_query', '')
        conditions = []

        gpa_match = re.search(r'gpa\s+(above|greater than|>\s*|below|less than|<\s*)\s*([\d.]+)', query.lower())
        if gpa_match:
            op = '>' if gpa_match.group(1) in ['above', 'greater than', '>'] else '<'
            conditions.append(f"overall_gpa {op} {float(gpa_match.group(2))}")

        att_match = re.search(r'attendance\s+(above|greater than|>\s*|below|less than|<\s*)\s*(\d+)%', query.lower())
        if att_match:
            op = '>' if att_match.group(1) in ['above', 'greater than', '>'] else '<'
            conditions.append(f"CAST(a.attendance AS REAL) {op} {float(att_match.group(2))}")

        section_match = re.search(r'section\s+([A-Z])', query, re.IGNORECASE)
        if section_match:
            conditions.append(f"UPPER(sec.title) = '{section_match.group(1).upper()}'")

        if not conditions:
            return self._error_response("No complex filter conditions found")

        where_clause = " AND ".join(conditions)
        sql = f"""
        SELECT s.name, s.uuid,
               sec.title AS section,
               a.attendance,
               ROUND(SUM(CAST(m.mark AS REAL)) * 100.0 / SUM(CAST(ep.fullMark AS REAL)), 2) AS overall_percentage,
               GET_GPA(SUM(CAST(m.mark AS REAL)) * 100.0 / SUM(CAST(ep.fullMark AS REAL))) AS overall_gpa,
               GET_GRADE(SUM(CAST(m.mark AS REAL)) * 100.0 / SUM(CAST(ep.fullMark AS REAL))) AS overall_grade
        FROM students s
        JOIN marks m ON s.uuid = m.student_id
        JOIN exam_parameters ep ON m.exam_parameter_id = ep.id
        JOIN sections sec ON s.section_id = sec.id
        JOIN attendances a ON s.id = a.student_id
        GROUP BY s.uuid, s.name, sec.title, a.attendance
        HAVING {where_clause}
        ORDER BY overall_gpa DESC
        """
        try:
            df = self.sql_gen.execute_query(sql)
            return ResultFormatter.format(
                query=parsed['original_query'],
                result_df=df,
                query_type='complex_filter',
                entities=parsed['entities'],
                sql=sql
            )
        except Exception as e:
            return self._error_response(f"Error processing complex filter: {e}")

    def _error_response(self, message: str) -> Dict:
        return {
            'success': False,
            'error': message,
            'formatted': f"? {message}",
            'interpretation': f"**1. Error**\n\n{message}",
            'result_df': [],
            'rows': 0
        }

    def _general_response(self, query: str, error: str = None) -> Dict:
        return {
            'success': True if not error else False,
            'query': query,
            'formatted': f"?? I understood your query as: '{query}'\n\nPlease be more specific." + (f"\n\nError: {error}" if error else ""),
            'interpretation': "**1. Overview**\n\nI couldn't determine the specific intent. Please refine your query." + (f"\n\nError: {error}" if error else ""),
            'result_df': [],
            'rows': 0
        }


# =============================================================================
# API ROUTES
# =============================================================================

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    session_id = get_session_id(request)
    if not session_id:
        session_id = session_manager.create_session()
    response = templates.TemplateResponse("index.html", {"request": request})
    create_session_response(response, session_id)
    return response


@app.get("/upload", response_class=HTMLResponse)
async def upload_page(request: Request):
    session_id = get_session_id(request)
    if not session_id:
        session_id = session_manager.create_session()
    response = templates.TemplateResponse("upload.html", {"request": request})
    create_session_response(response, session_id)
    return response


@app.get("/query", response_class=HTMLResponse)
async def query_page(request: Request):
    session_id = get_session_id(request)
    if not session_id:
        session_id = session_manager.create_session()
    response = templates.TemplateResponse("query.html", {"request": request})
    create_session_response(response, session_id)
    return response


@app.get("/schema", response_class=HTMLResponse)
async def schema_page(request: Request):
    session_id = get_session_id(request)
    if not session_id:
        session_id = session_manager.create_session()
    response = templates.TemplateResponse("schema.html", {"request": request})
    create_session_response(response, session_id)
    return response


@app.get("/data", response_class=HTMLResponse)
async def data_page(request: Request):
    session_id = get_session_id(request)
    if not session_id:
        session_id = session_manager.create_session()
    response = templates.TemplateResponse("data.html", {"request": request})
    create_session_response(response, session_id)
    return response


@app.post("/api/upload")
async def save_chat_json(request: Request, payload: SaveChatPayload):
    try:
        session_id = get_session_id(request)
        if not session_id:
            session_id = session_manager.create_session()

        base_name = os.path.basename(payload.fileName)
        clean_name = os.path.splitext(base_name)[0]
        filename = f"{clean_name}.json"
        file_path = os.path.join(UPLOAD_DIR, filename)
        os.makedirs(UPLOAD_DIR, exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(payload.json, f, indent=2, ensure_ascii=False)

        tables, rel_map = read_dataset_file(file_path)
        total_rows = sum(len(df) for df in tables.values())

        session_manager.set_dataset(session_id, clean_name, tables, rel_map)

        response_data = {
            "success": True,
            "message": f"Saved '{filename}' successfully",
            "dataset_name": clean_name,
            "tables": list(tables.keys()),
            "total_rows": total_rows,
            "relationship_count": len(rel_map.relationships)
        }

        response = JSONResponse(content=response_data)
        create_session_response(response, session_id)
        return response

    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/upload_old")
async def upload_file(request: Request, file: UploadFile = File(...)):
    try:
        session_id = get_session_id(request)
        if not session_id:
            session_id = session_manager.create_session()

        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        tables, rel_map = read_dataset_file(file_path)
        dataset_name = os.path.splitext(file.filename)[0]

        total_rows = sum(len(df) for df in tables.values())

        session_manager.set_dataset(session_id, dataset_name, tables, rel_map)

        response_data = {
            "success": True,
            "message": f"Uploaded '{dataset_name}' successfully",
            "dataset_name": dataset_name,
            "tables": list(tables.keys()),
            "total_rows": total_rows,
            "relationship_count": len(rel_map.relationships)
        }

        response = JSONResponse(content=response_data)
        create_session_response(response, session_id)
        return response
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/datasets")
async def list_datasets(request: Request):
    session_id = get_session_id(request)
    if not session_id:
        return {"datasets": []}
    datasets = []
    for name in session_manager.list_datasets(session_id):
        datasets.append({"name": name, "file": f"{name}.json"})
    return {"datasets": datasets}


@app.get("/api/dataset/{dataset_name}")
async def get_dataset(request: Request, dataset_name: str):
    session_id = get_session_id(request)
    if not session_id:
        raise HTTPException(status_code=401, detail="No session found")
    tables, rel_map = session_manager.get_dataset(session_id, dataset_name)
    if not tables:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return {
        "name": dataset_name,
        "tables": {name: {"columns": list(df.columns), "rows": len(df)} for name, df in tables.items()},
        "relationships": rel_map.to_dict() if rel_map else {}
    }


@app.post("/api/query")
async def query_data(request: Request, query_req: QueryRequest):
    try:
        session_id = get_session_id(request)
        if not session_id:
            session_id = session_manager.create_session()

        tables, rel_map = session_manager.get_dataset(session_id, query_req.dataset_name)
        if not tables:
            file_path = os.path.join(UPLOAD_DIR, f"{query_req.dataset_name}.json")
            if not os.path.exists(file_path):
                file_path = os.path.join(UPLOAD_DIR, f"{query_req.dataset_name}.csv")
            if not os.path.exists(file_path):
                return {"success": False, "error": f"Dataset '{query_req.dataset_name}' not found"}
            tables, rel_map = read_dataset_file(file_path)

        handler = UniversalQueryHandler(tables, rel_map)
        result = handler.handle_query(query_req.query)

        parsed = UniversalSemanticQueryParser.parse(query_req.query)
        result['query_type'] = parsed.get('query_type', 'general')
        result['parsed_entities'] = parsed.get('entities', {})

        session_manager.add_to_history(session_id, query_req.query, result)

        return result
    except Exception as e:
        logger.error(f"Query error: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


@app.get("/api/history")
async def get_history(request: Request):
    session_id = get_session_id(request)
    if not session_id:
        return {"history": []}
    return {"history": session_manager.get_history(session_id)}


@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "active_sessions": session_manager.get_session_count()
    }


@app.post("/api/session/logout")
async def logout(request: Request, response: Response):
    session_id = request.cookies.get("session_id")
    if session_id:
        session_manager.delete_session(session_id)
    response = JSONResponse({"success": True, "message": "Logged out successfully"})
    response.delete_cookie("session_id", path="/")
    return response


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8005,
        reload=False
    )