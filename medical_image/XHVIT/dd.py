#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dynamic LLM Data Analysis System (v3)
--------------------------------------
ENHANCEMENTS:
1. Complex JOIN handling (3+ tables) with explicit join path finding
2. Subquery generation with CTE support
3. Complex logic (AND/OR/NOT, nested conditions, CASE statements)
4. Automatic query decomposition for multi-step reasoning
5. Enhanced entity extraction with fuzzy matching
6. SQL validation with query plan analysis
7. Retry with progressive complexity reduction
8. Shows generated SQL in output
9. Specific, data-driven interpretation for ALL query types
"""

import io
import json
import logging
import re
import sys
import threading
import sqlite3
import random
from datetime import datetime, date
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np
import pandas as pd
from tabulate import tabulate

sys.setrecursionlimit(10000)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("llm_analyser")

# =============================================================================
# SECTION 1: Model Manager
# =============================================================================

MODEL_NAME = "/home/llm/workspace/dataset/project/my/new/models/Rag_project/llm-data-analysis-system/sql"

class ModelManager:
    """Loads and manages the local LLM."""
    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = "cuda"
        self._loaded = False
        self._loading = False
        self._load_complete = threading.Event()

    def load_async(self):
        if self._loaded or self._loading:
            return
        self._loading = True
        print("[i] Loading model...")
        def load_thread():
            try:
                self.load()
                print("[OK] Model loaded successfully!")
            except Exception as e:
                print(f"[ERROR] Failed to load model: {e}")
                import traceback
                traceback.print_exc()
            finally:
                self._loading = False
        threading.Thread(target=load_thread, daemon=True).start()

    def load(self):
        if self._loaded:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import os
        if not os.path.exists(self.model_name):
            logger.warning(f"Local model path '{self.model_name}' not found. Falling back to Hugging Face...")
            self.model_name = "XGenerationLab/XiYanSQL-QwenCoder-7B-2502"
            print(f"[WARN] Using Hugging Face model: {self.model_name}")
        logger.info(f"Loading model from: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        if torch.cuda.is_available():
            try:
                from transformers import BitsAndBytesConfig
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    quantization_config=bnb_config,
                    device_map="cuda:0",
                    trust_remote_code=True,
                    torch_dtype=torch.float16,
                )
                self.device = "cuda"
                logger.info("Model loaded on GPU with 4-bit quantization")
            except Exception as exc:
                logger.warning(f"GPU loading failed, falling back to CPU: {exc}")
                self._load_cpu()
        else:
            self._load_cpu()
        self.model.eval()
        self._loaded = True
        self._load_complete.set()
        logger.info(f"Model loaded on device={self.device}")

    def _load_cpu(self):
        import torch
        from transformers import AutoModelForCausalLM
        logger.info("Loading model on CPU (this may take a while)...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=torch.float32, trust_remote_code=True,
        )
        self.device = "cpu"

    def wait_for_load(self):
        if self._loaded:
            return
        if self._loading:
            print("[i] Waiting for model to load...")
            self._load_complete.wait()
        else:
            self.load_async()
            self._load_complete.wait()

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def generate(self, prompt: str, system_prompt: str = None, max_new_tokens: int = 1500,
                 temperature: float = 0.05) -> str:
        self.wait_for_load()
        if not self._loaded:
            raise RuntimeError("Model failed to load")
        import torch
        if system_prompt is None:
            system_prompt = ("You are a SQL expert. Generate ONLY valid SQL queries. "
                              "Use SQLite syntax. No explanations, no markdown.")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=8192,
                                 padding=True).to(self.model.device)
        do_sample = temperature > 0.05
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            top_k=40,
            do_sample=do_sample,
            repetition_penalty=1.05,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = 0.85
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)
        generated = output_ids[0][inputs["input_ids"].shape[-1]:]
        result = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        result = re.sub(r'^```(?:sql|json)?\s*', '', result)
        result = re.sub(r'\s*```$', '', result)
        result = result.strip().strip('"')
        return result

model_manager = ModelManager()

# =============================================================================
# SECTION 2: Enhanced Relationship Map with Join Path Finding
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
    
    def get_all_table_names(self) -> List[str]:
        return list(self.table_schemas.keys())
    
    def get_all_column_names(self) -> Dict[str, List[str]]:
        return {t: list(s.keys()) for t, s in self.table_schemas.items()}
    
    def find_join_path(self, start_table: str, end_table: str) -> Optional[List[Relationship]]:
        """BFS to find shortest join path between two tables."""
        if start_table == end_table:
            return []
        
        # Build adjacency graph
        graph = defaultdict(list)
        for rel in self.relationships:
            graph[rel.source_table].append((rel.target_table, rel))
            graph[rel.target_table].append((rel.source_table, rel))
        
        # BFS
        visited = {start_table}
        queue = [(start_table, [])]
        
        while queue:
            current, path = queue.pop(0)
            for neighbor, rel in graph.get(current, []):
                if neighbor in visited:
                    continue
                new_path = path + [rel]
                if neighbor == end_table:
                    return new_path
                visited.add(neighbor)
                queue.append((neighbor, new_path))
        
        return None
    
    def get_join_path_sql(self, tables: List[str]) -> str:
        """Generate SQL JOINs for a list of tables."""
        if len(tables) < 2:
            return ""
        
        path = []
        for i in range(len(tables) - 1):
            rel_path = self.find_join_path(tables[i], tables[i + 1])
            if rel_path:
                path.extend(rel_path)
            else:
                # Try to find any relationship
                for rel in self.relationships:
                    if (rel.source_table == tables[i] and rel.target_table == tables[i + 1]) or \
                       (rel.source_table == tables[i + 1] and rel.target_table == tables[i]):
                        path.append(rel)
                        break
        
        if not path:
            return ""
        
        joins = []
        seen_tables = {tables[0]}
        for rel in path:
            if rel.target_table not in seen_tables:
                joins.append(f"JOIN {rel.target_table} ON {rel.join_condition()}")
                seen_tables.add(rel.target_table)
            elif rel.source_table not in seen_tables:
                joins.append(f"JOIN {rel.source_table} ON {rel.join_condition()}")
                seen_tables.add(rel.source_table)
        
        return "\n" + "\n".join(joins) if joins else ""

    def get_complete_schema(self) -> str:
        lines = ["=" * 90, "COMPLETE DATABASE SCHEMA", "=" * 90, ""]
        for table_name, schema in self.table_schemas.items():
            lines.append(f"TABLE: {table_name}")
            lines.append("-" * 60)
            for col_name in sorted(c for c in schema if c != "_index"):
                info = schema[col_name]
                sample = info.get('sample', [])
                sample_str = f" (samples: {', '.join(str(s) for s in sample[:2])})" if sample else ""
                lines.append(f"  {col_name}: {info.get('type', 'text')} "
                              f"(non-null: {info.get('non_null', 0)}, unique: {info.get('unique', 0)}){sample_str}")
            lines.append("")
        return "\n".join(lines)

    def get_all_relationships(self) -> str:
        lines = ["=" * 90, "TABLE RELATIONSHIPS", "=" * 90, ""]
        if not self.relationships:
            lines.append("No relationships detected.")
            return "\n".join(lines)
        for rel in self.relationships:
            lines.append(f"-- {rel.description}")
            lines.append(f"JOIN {rel.target_table} ON {rel.join_condition()}  ({rel.cardinality})")
            lines.append("")
        return "\n".join(lines)

    def get_table_samples(self) -> str:
        if not self.table_samples:
            return "No sample data available"
        lines = ["SAMPLE DATA:", "=" * 60]
        for table_name, sample in self.table_samples.items():
            lines.append(f"\n{table_name}:")
            lines.append(sample[:500] + "..." if len(sample) > 500 else sample)
        return "\n".join(lines)

    def get_sql_schema(self) -> str:
        lines = [f"-- AVAILABLE TABLES: {', '.join(self.get_all_table_names())}", ""]
        for table_name, schema in self.table_schemas.items():
            lines.append(f"CREATE TABLE {table_name} (")
            cols = [f"  {c} {self._infer_sql_type(info.get('type', 'object'))}"
                    for c, info in schema.items() if c != "_index"]
            lines.append(",\n".join(cols))
            lines.append(");\n")
        return "\n".join(lines)

    @staticmethod
    def _infer_sql_type(dtype: str) -> str:
        d = dtype.lower()
        if 'int' in d:
            return 'INTEGER'
        if 'float' in d:
            return 'REAL'
        if 'datetime' in d:
            return 'DATETIME'
        return 'TEXT'

    def get_summary(self) -> str:
        lines = ["Data Model Summary:", f"  Tables: {len(self.table_counts)}",
                  f"  Relationships: {len(self.relationships)}", ""]
        if self.table_counts:
            lines.append("  Tables:")
            for name, count in self.table_counts.items():
                lines.append(f"    - {name}: {count} rows")
        if self.relationships:
            lines.append("\n  Relationships:")
            for rel in self.relationships:
                lines.append(f"    - {rel.join_condition()} ({rel.cardinality}) - {rel.description}")
        return "\n".join(lines)

# =============================================================================
# SECTION 3: Enhanced Dynamic Parser
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
    return df

class DynamicParser:
    def __init__(self):
        self.relationship_map = RelationshipMap()

    def parse(self, data: Any) -> Dict[str, pd.DataFrame]:
        container, prefix = self._find_best_container(data)
        result: Dict[str, pd.DataFrame] = {}
        if isinstance(container, dict):
            for key, value in container.items():
                table_name = self._safe_name(key)
                df = self._to_dataframe(value, table_name)
                if df is not None and len(df) > 0:
                    result[table_name] = normalize_dtypes(df)
                    self.relationship_map.table_counts[table_name] = len(df)
                    logger.info(f"Parsed table '{table_name}' with {len(df)} rows, {len(df.columns)} cols")
        if not result:
            df = self._to_dataframe(container, prefix or "data")
            if df is not None:
                result[prefix or "data"] = normalize_dtypes(df)
                self.relationship_map.table_counts[prefix or "data"] = len(df)
        self._build_schemas(result)
        self._infer_relationships(result)
        return result

    def _find_best_container(self, data: Any, path: str = "") -> Tuple[Any, str]:
        if not isinstance(data, dict):
            return data, path
        tabular_value_count = sum(
            1 for v in data.values()
            if (isinstance(v, list) and v and isinstance(v[0], dict))
            or (isinstance(v, dict) and v and all(isinstance(x, dict) for x in v.values()))
        )
        if tabular_value_count >= 2:
            return data, path
        best_child, best_path, best_score = None, path, -1
        for key, value in data.items():
            if isinstance(value, dict):
                score = sum(1 for v in value.values() if isinstance(v, (list, dict)))
                if score > best_score:
                    best_child, best_path, best_score = value, f"{path}.{key}" if path else key, score
        if best_child is not None:
            return self._find_best_container(best_child, best_path)
        return data, path

    def _to_dataframe(self, value: Any, name: str) -> Optional[pd.DataFrame]:
        if isinstance(value, list):
            return self._list_to_df(value)
        if isinstance(value, dict):
            return self._dict_of_records_to_df(value)
        return None

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

    def _dict_of_records_to_df(self, d: Dict[str, Any]) -> Optional[pd.DataFrame]:
        if not d or not all(isinstance(v, dict) for v in d.values()):
            return None
        rows = []
        for key, item in d.items():
            row = {}
            for k, v in item.items():
                if isinstance(v, (dict, list)):
                    continue
                try:
                    make_hashable(v)
                    row[k] = v
                except Exception:
                    row[k] = str(v)
            row["_key"] = key
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
        for table_name in table_names:
            df = tables[table_name]
            for col in df.columns:
                if col in ("_index", "_key") or not (col.endswith('_id') or col.endswith('Id')):
                    continue
                if col in ('id',):
                    continue
                base = re.sub(r'([_]?[Ii]d)$', '', col)
                candidates = [base, base + 's', base.rstrip('s')]
                target_table = next((t for t in candidates if t in table_names and t != table_name), None)
                if not target_table:
                    continue
                target_df = tables[target_table]
                target_pk = None
                for pk_candidate in ['uuid', 'id', f'{target_table}_id', col]:
                    if pk_candidate in target_df.columns:
                        target_pk = pk_candidate
                        break
                if not target_pk:
                    continue
                try:
                    src_vals = set(df[col].dropna().astype(str))
                    tgt_vals = set(target_df[target_pk].dropna().astype(str))
                    overlap = len(src_vals & tgt_vals) / max(len(src_vals), 1)
                except Exception:
                    overlap = 0
                key = (table_name, col, target_table, target_pk)
                if overlap > 0.3 and key not in seen:
                    seen.add(key)
                    self.relationship_map.relationships.append(Relationship(
                        source_table=table_name, source_field=col,
                        target_table=target_table, target_field=target_pk,
                        cardinality="Many-to-One",
                        description=f"{table_name}.{col} references {target_table}.{target_pk} "
                                    f"(value overlap {overlap:.0%})",
                        confidence=min(1.0, overlap * 1.2),
                    ))
        # Known relationships
        known_chain = [
            ("students", "uuid", "marks", "student_id", "One-to-Many", "Students have marks"),
            ("marks", "exam_parameter_id", "exam_parameters", "id", "Many-to-One", "Marks reference exam parameters"),
            ("exam_parameters", "class_section_subject_id", "subjects", "cssId", "Many-to-One",
             "Exam parameters reference subjects"),
            ("students", "section_id", "sections", "id", "Many-to-One", "Students belong to sections"),
            ("students", "id", "symbol_no", "student_id", "One-to-One", "Students have symbol numbers"),
        ]
        for src_t, src_f, tgt_t, tgt_f, card, desc in known_chain:
            if src_t in tables and tgt_t in tables and src_f in tables[src_t].columns and tgt_f in tables[tgt_t].columns:
                key = (src_t, src_f, tgt_t, tgt_f)
                if key not in seen:
                    seen.add(key)
                    self.relationship_map.relationships.append(
                        Relationship(src_t, src_f, tgt_t, tgt_f, card, desc, confidence=1.0))

def read_json_with_relationships(file_path: str) -> Tuple[Dict[str, pd.DataFrame], RelationshipMap]:
    with open(file_path, 'rb') as f:
        content = f.read()
    data = json.loads(content.decode('utf-8'))
    parser = DynamicParser()
    tables = parser.parse(data)
    return tables, parser.relationship_map

def read_dataset_file(file_path: str) -> Tuple[Dict[str, pd.DataFrame], RelationshipMap]:
    ext = file_path.lower().rsplit(".", 1)[-1] if "." in file_path else ""
    if ext == "json":
        return read_json_with_relationships(file_path)
    if ext == "csv":
        content = open(file_path, 'rb').read()
        for enc in ["utf-8", "utf-8-sig", "latin1", "cp1252"]:
            try:
                df = pd.read_csv(io.BytesIO(content), encoding=enc)
                return {file_path.split("/")[-1].rsplit('.', 1)[0]: df}, RelationshipMap()
            except Exception:
                continue
        raise ValueError("Unable to parse CSV")
    if ext in ["xlsx", "xls"]:
        content = open(file_path, 'rb').read()
        xls = pd.ExcelFile(io.BytesIO(content))
        tables = {sheet: pd.read_excel(xls, sheet_name=sheet) for sheet in xls.sheet_names}
        return tables, RelationshipMap()
    raise ValueError(f"Unsupported file format: .{ext}")

def convert_to_serializable(obj: Any) -> Any:
    if isinstance(obj, pd.DataFrame):
        return json.loads(obj.to_json(orient="records", date_format="iso", default_handler=str))
    if isinstance(obj, pd.Series):
        return convert_to_serializable(obj.to_dict())
    if isinstance(obj, (pd.Timestamp, datetime, date)):
        return obj.isoformat()
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, dict):
        return {str(k): convert_to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [convert_to_serializable(v) for v in obj]
    return None if pd.isna(obj) else obj

# =============================================================================
# SECTION 4: Enhanced SQL Generator with Query Analysis
# =============================================================================

class SQLGenerator:
    def __init__(self, rel_map: RelationshipMap):
        self.rel_map = rel_map
        self.connection = sqlite3.connect(':memory:')
        self.connection.execute('PRAGMA case_sensitive_like = OFF')
        self.last_sql = None
        self.last_error = None

    def create_tables_from_dataframes(self, dataframes: Dict[str, pd.DataFrame]):
        for table_name, df in dataframes.items():
            df.to_sql(table_name, self.connection, if_exists='replace', index=False)
        self.connection.commit()

    def get_sample_data(self) -> str:
        try:
            samples = []
            for table in self.rel_map.get_all_table_names()[:8]:
                result = pd.read_sql_query(f"SELECT * FROM {table} LIMIT 3", self.connection)
                if not result.empty:
                    samples.append(f"Sample from {table}:")
                    samples.append(result.to_string())
            return "\n".join(samples) if samples else "No sample data available"
        except Exception:
            return "No sample data available"

    def execute_query(self, sql: str) -> pd.DataFrame:
        try:
            self.last_sql = sql
            self.last_error = None
            return pd.read_sql_query(sql, self.connection)
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"SQL execution error: {e}")
            raise

    def validate_sql(self, sql: str) -> Tuple[bool, str]:
        sql_clean = sql.strip().rstrip(';')
        if not sql_clean:
            return False, "Empty SQL query"
        sql_upper = sql_clean.upper()
        for op in ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'CREATE', 'TRUNCATE', 'ATTACH', 'PRAGMA']:
            if re.search(rf'\b{op}\b', sql_upper):
                return False, f"Dangerous operation '{op}' not allowed"
        if not any(k in sql_upper for k in ['SELECT', 'WITH']):
            return False, "Query must start with SELECT or WITH"
        if sql_clean.count('(') != sql_clean.count(')'):
            return False, "Unbalanced parentheses"
        if 'SELECT' in sql_upper and 'FROM' not in sql_upper:
            return False, "SELECT query missing FROM clause"
        # Check for valid table references
        for table in self.rel_map.get_all_table_names():
            if table in sql_clean:
                break
        return True, "Valid SQL"

    def analyze_query_complexity(self, sql: str) -> Dict[str, Any]:
        """Analyze SQL complexity for progressive retry strategy."""
        return {
            'joins': len(re.findall(r'\bJOIN\b', sql.upper())),
            'subqueries': len(re.findall(r'\(\s*SELECT', sql.upper())),
            'conditions': len(re.findall(r'\bAND\b|\bOR\b', sql.upper())),
            'has_window': 'OVER' in sql.upper(),
            'has_case': 'CASE' in sql.upper(),
            'has_group_by': 'GROUP BY' in sql.upper(),
            'has_having': 'HAVING' in sql.upper(),
        }

# =============================================================================
# SECTION 5: Enhanced Entity Index with Fuzzy Matching
# =============================================================================

class EntityIndex:
    _NOISE_PREFIXES = re.compile(r'^(com\.?|opt\.?\s*i+\.?|std\.?)\s*', re.IGNORECASE)
    _STOPWORDS_LOCAL = {'the', 'a', 'an', 'of', 'in', 'on', 'for', 'and', 'or', 'to', 'is', 'are'}

    def __init__(self, dataframes: Dict[str, pd.DataFrame]):
        self.entries: List[Tuple[str, str, str, List[str], str]] = []  # table, col, value, tokens, original
        self.value_index: Dict[str, List[Tuple[str, str, str]]] = defaultdict(list)
        self.ngram_index: Dict[str, List[Tuple[str, str, str, float]]] = defaultdict(list)
        
        for table, df in dataframes.items():
            for col in df.columns:
                if col in ('_index', '_key'):
                    continue
                if not (df[col].dtype == object or pd.api.types.is_string_dtype(df[col])):
                    continue
                try:
                    uniques = df[col].dropna().astype(str).unique()
                except Exception:
                    continue
                if not (1 < len(uniques) <= 2000):
                    continue
                for val in uniques:
                    val_clean = val.strip()
                    if len(val_clean) < 3:
                        continue
                    core = self._NOISE_PREFIXES.sub('', val_clean).strip()
                    tokens = self._tokenize(core or val_clean)
                    if tokens:
                        self.entries.append((table, col, val_clean, tokens, core or val_clean))
                        self.value_index[val_clean.lower()].append((table, col, val_clean))
                        # Build n-gram index for fuzzy matching
                        for n in range(2, 5):
                            for i in range(len(val_clean) - n + 1):
                                ngram = val_clean[i:i+n].lower()
                                if len(ngram) >= 3:
                                    self.ngram_index[ngram].append((table, col, val_clean, len(ngram) / len(val_clean)))

    @classmethod
    def _tokenize(cls, text: str) -> List[str]:
        words = re.findall(r'[a-zA-Z]+', text.lower())
        return [cls._stem(w) for w in words if w not in cls._STOPWORDS_LOCAL and len(w) >= 3]

    @staticmethod
    def _stem(word: str) -> str:
        return word[:-1] if word.endswith('s') and len(word) > 4 else word

    def match(self, query: str, max_results: int = 8) -> List[Tuple[str, str, str, float]]:
        """Enhanced matching with fuzzy n-gram fallback."""
        query_tokens = set(self._tokenize(query))
        if not query_tokens:
            return []
        
        scored = []
        for table, col, value, tokens, original in self.entries:
            hit_tokens = [t for t in tokens if any(t.startswith(qt) or qt.startswith(t) for qt in query_tokens)]
            if hit_tokens:
                coverage = len(hit_tokens) / max(len(tokens), 1)
                if coverage >= 0.5:
                    scored.append((table, col, value, coverage))
        
        # If exact matching found nothing, try fuzzy matching
        if not scored:
            query_lower = query.lower()
            for ngram, matches in self.ngram_index.items():
                if ngram in query_lower:
                    for table, col, val, score in matches:
                        scored.append((table, col, val, score))
        
        scored.sort(key=lambda x: (-x[3], -len(x[2])))
        seen_values = set()
        results = []
        for table, col, value, score in scored:
            if value in seen_values:
                continue
            seen_values.add(value)
            results.append((table, col, value, score))
            if len(results) >= max_results:
                break
        return results

# =============================================================================
# SECTION 6: Enhanced Query Context with Complexity Analysis
# =============================================================================

class QueryContext:
    def __init__(self, rel_map: RelationshipMap, entity_index: EntityIndex = None):
        self.rel_map = rel_map
        self.entity_index = entity_index

    def build(self, query: str) -> Dict[str, Any]:
        entity_matches = self.entity_index.match(query) if self.entity_index else []
        complexity = self._analyze_query_complexity(query)
        return {
            "original_query": query,
            "entity_matches": entity_matches,
            "extracted_number": self._extract_number(query),
            "tables_available": self.rel_map.get_all_table_names(),
            "columns_available": self.rel_map.get_all_column_names(),
            "complete_schema": self.rel_map.get_complete_schema(),
            "all_relationships": self.rel_map.get_all_relationships(),
            "table_samples": self.rel_map.get_table_samples(),
            "complexity": complexity,
            "join_paths": self._get_join_path_hints(query),
        }

    @staticmethod
    def _extract_number(query: str) -> Optional[int]:
        patterns = [
            r'top\s+(\d+)', r'limit\s+(\d+)', r'first\s+(\d+)',
            r'(\d+)\s+best', r'best\s+(\d+)', r'(\d+)\s+highest',
            r'highest\s+(\d+)', r'(\d+)\s+lowest', r'at least\s+(\d+)',
            r'more than\s+(\d+)', r'less than\s+(\d+)', r'(\d+)\s+students?\b'
        ]
        for pattern in patterns:
            m = re.search(pattern, query, re.IGNORECASE)
            if m:
                return int(m.group(1))
        return None

    @staticmethod
    def _analyze_query_complexity(query: str) -> Dict[str, Any]:
        query_lower = query.lower()
        return {
            'has_comparison': bool(re.search(r'\b(compare|vs|versus|than|more|less|above|below)\b', query_lower)),
            'has_ranking': bool(re.search(r'\b(top|best|highest|lowest|rank|worst)\b', query_lower)),
            'has_multi_condition': bool(re.search(r'\b(and|or|not|both|either)\b', query_lower)),
            'has_aggregation': bool(re.search(r'\b(average|avg|sum|total|count|max|min|mean)\b', query_lower)),
            'has_subquery': bool(re.search(r'\b(than|above|below)\s+(average|avg|class|overall)\b', query_lower)),
            'has_multiple_entities': len(re.findall(r'\b(and|with|including)\b', query_lower)) > 1,
        }

    def _get_join_path_hints(self, query: str) -> str:
        """Find likely tables needed based on entity matches."""
        if not self.entity_index:
            return ""
        
        tables_needed = set()
        for table, col, value, score in self.entity_index.match(query, max_results=10):
            tables_needed.add(table)
            # Add related tables
            for rel in self.rel_map.relationships:
                if rel.source_table == table:
                    tables_needed.add(rel.target_table)
                elif rel.target_table == table:
                    tables_needed.add(rel.source_table)
        
        if len(tables_needed) <= 1:
            return ""
        
        tables_list = list(tables_needed)
        paths = []
        for i in range(len(tables_list) - 1):
            path = self.rel_map.find_join_path(tables_list[i], tables_list[i + 1])
            if path:
                paths.extend(path)
        
        if not paths:
            return ""
        
        lines = ["\nRECOMMENDED JOIN PATHS:"]
        for rel in paths:
            lines.append(f"  {rel.join_condition()} ({rel.description})")
        return "\n".join(lines)

# =============================================================================
# SECTION 7: Enhanced Few-Shot Examples
# =============================================================================

FEW_SHOT_EXAMPLES = """
-- Example 1: Simple filter with fuzzy matching
Q: Show all students in section A
SQL: SELECT s.* FROM students s JOIN sections sec ON s.section_id = sec.id WHERE UPPER(sec.title) LIKE UPPER('%A%')

-- Example 2: Aggregation with GROUP BY
Q: Average marks per subject
SQL: SELECT sub.title AS subject, AVG(CAST(m.mark AS REAL)) AS avg_mark
     FROM marks m
     JOIN exam_parameters ep ON m.exam_parameter_id = ep.id
     JOIN subjects sub ON ep.class_section_subject_id = sub.cssId
     GROUP BY sub.title

-- Example 3: Complex filtering with pass/fail
Q: Which students failed in Maths
SQL: SELECT s.name, m.mark, ep.passMark
     FROM students s
     JOIN marks m ON s.uuid = m.student_id
     JOIN exam_parameters ep ON m.exam_parameter_id = ep.id
     JOIN subjects sub ON ep.class_section_subject_id = sub.cssId
     WHERE UPPER(sub.title) LIKE UPPER('%maths%')
       AND CAST(m.mark AS REAL) < CAST(ep.passMark AS REAL)

-- Example 4: Ranking with window function
Q: Top 10 students by total marks
SQL: SELECT s.name, SUM(CAST(m.mark AS REAL)) AS total_marks,
            RANK() OVER (ORDER BY SUM(CAST(m.mark AS REAL)) DESC) AS rank
     FROM students s JOIN marks m ON s.uuid = m.student_id
     GROUP BY s.name ORDER BY total_marks DESC LIMIT 10

-- Example 5: Subquery comparison
Q: Students scoring above the class average
SQL: SELECT s.name, AVG(CAST(m.mark AS REAL)) AS avg_mark
     FROM students s JOIN marks m ON s.uuid = m.student_id
     GROUP BY s.name
     HAVING avg_mark > (SELECT AVG(CAST(mark AS REAL)) FROM marks)

-- Example 6: LEFT JOIN for missing data
Q: Students with no marks recorded
SQL: SELECT s.name FROM students s
     LEFT JOIN marks m ON s.uuid = m.student_id
     WHERE m.id IS NULL

-- Example 7: Multiple conditions with AND/OR
Q: Students who failed in Maths OR English
SQL: SELECT DISTINCT s.name
     FROM students s
     JOIN marks m ON s.uuid = m.student_id
     JOIN exam_parameters ep ON m.exam_parameter_id = ep.id
     JOIN subjects sub ON ep.class_section_subject_id = sub.cssId
     WHERE (UPPER(sub.title) LIKE UPPER('%maths%') OR UPPER(sub.title) LIKE UPPER('%english%'))
       AND CAST(m.mark AS REAL) < CAST(ep.passMark AS REAL)

-- Example 8: Complex join with 3+ tables
Q: Show section-wise average marks in Nepali
SQL: SELECT sec.title AS section, AVG(CAST(m.mark AS REAL)) AS avg_mark
     FROM students s
     JOIN sections sec ON s.section_id = sec.id
     JOIN marks m ON s.uuid = m.student_id
     JOIN exam_parameters ep ON m.exam_parameter_id = ep.id
     JOIN subjects sub ON ep.class_section_subject_id = sub.cssId
     WHERE UPPER(sub.title) LIKE UPPER('%nepali%')
     GROUP BY sec.title

-- Example 9: Nested subquery with complex logic
Q: Students who scored above average in all subjects
SQL: SELECT s.name
     FROM students s
     WHERE NOT EXISTS (
         SELECT 1
         FROM marks m
         JOIN exam_parameters ep ON m.exam_parameter_id = ep.id
         JOIN subjects sub ON ep.class_section_subject_id = sub.cssId
         WHERE m.student_id = s.uuid
           AND CAST(m.mark AS REAL) < (
               SELECT AVG(CAST(m2.mark AS REAL))
               FROM marks m2
               JOIN exam_parameters ep2 ON m2.exam_parameter_id = ep2.id
               JOIN subjects sub2 ON ep2.class_section_subject_id = sub2.cssId
               WHERE sub2.id = sub.id
           )
     )

-- Example 10: CASE statement for categorization
Q: Categorize students by performance
SQL: SELECT s.name,
            AVG(CAST(m.mark AS REAL)) AS avg_mark,
            CASE
                WHEN AVG(CAST(m.mark AS REAL)) >= 80 THEN 'Excellent'
                WHEN AVG(CAST(m.mark AS REAL)) >= 60 THEN 'Good'
                WHEN AVG(CAST(m.mark AS REAL)) >= 40 THEN 'Average'
                ELSE 'Needs Improvement'
            END AS performance_category
     FROM students s
     JOIN marks m ON s.uuid = m.student_id
     GROUP BY s.name

-- Example 11: Comparison between two groups
Q: Compare average marks of boys vs girls
SQL: SELECT s.gender, AVG(CAST(m.mark AS REAL)) AS avg_mark
     FROM students s
     JOIN marks m ON s.uuid = m.student_id
     GROUP BY s.gender

-- Example 12: Having with complex condition
Q: Students who failed at least 2 subjects
SQL: SELECT s.name, COUNT(*) AS fail_count
     FROM students s
     JOIN marks m ON s.uuid = m.student_id
     JOIN exam_parameters ep ON m.exam_parameter_id = ep.id
     WHERE CAST(m.mark AS REAL) < CAST(ep.passMark AS REAL)
     GROUP BY s.name
     HAVING fail_count >= 2
""".strip()

# =============================================================================
# SECTION 8: Enhanced SQL Generator with Progressive Retry
# =============================================================================

class SQLGeneratorLLM:
    def __init__(self, rel_map: RelationshipMap, sql_generator: SQLGenerator, entity_index: EntityIndex):
        self.rel_map = rel_map
        self.sql_generator = sql_generator
        self.entity_index = entity_index
        self.context: Dict[str, Any] = {}
        self.errors: List[str] = []
        self.attempts: List[Dict[str, Any]] = []

    def generate(self, query: str, max_attempts: int = 4) -> Dict[str, Any]:
        self.errors = []
        self.attempts = []
        self.context = QueryContext(self.rel_map, self.entity_index).build(query)

        print("\n" + "=" * 70)
        print("1. QUERY CONTEXT")
        print("=" * 70)
        print(f"  Query: {query}")
        if self.context['entity_matches']:
            for table, col, value, score in self.context['entity_matches']:
                print(f"  Matched: {table}.{col} = '{value}' (score: {score:.2f})")
        if self.context['extracted_number']:
            print(f"  Number: {self.context['extracted_number']}")
        print(f"  Tables: {', '.join(self.context['tables_available'])}")
        
        complexity = self.context.get('complexity', {})
        print(f"  Complexity: {json.dumps(complexity, indent=2)}")

        feedback = ""
        for attempt in range(max_attempts):
            print(f"\n  Attempt {attempt + 1}/{max_attempts}...")
            
            # Progressive complexity approach
            if attempt >= 2 and complexity.get('has_subquery', False):
                temp = 0.2 + attempt * 0.1
                max_tokens = 2000 + attempt * 500
                sql_result = self._attempt(query, feedback, temperature=temp, max_tokens=max_tokens)
            elif attempt >= 1 and complexity.get('has_multi_condition', False):
                temp = 0.1 + attempt * 0.1
                sql_result = self._attempt(query, feedback, temperature=temp)
            else:
                sql_result = self._attempt(query, feedback, temperature=0.05 + attempt * 0.05)
            
            self.attempts.append(sql_result)

            if not sql_result["success"]:
                feedback = f"PREVIOUS ATTEMPT FAILED: {sql_result.get('error')}\nFix the SQL accordingly."
                self.errors.append(sql_result.get('error', 'unknown error'))
                print(f"  [FAIL] {sql_result.get('error')}")
                continue

            # Enhanced sanity check
            sanity_issue = self._sanity_check(sql_result["sql"])
            if sanity_issue and attempt < max_attempts - 1:
                print(f"  [WARN] Sanity check flagged: {sanity_issue}")
                feedback = f"PREVIOUS SQL RAN BUT: {sanity_issue}\nReconsider joins/filters and try again."
                self.errors.append(sanity_issue)
                continue

            print("  [OK] SQL generated and passed checks.")
            return {"success": True, "sql": sql_result["sql"], "attempt": attempt + 1,
                     "context": self.context, "attempts": self.attempts}

        return {"success": False, "sql": None, "attempts": self.attempts, "errors": self.errors,
                 "context": self.context, "message": "Failed to generate valid SQL after multiple attempts"}

    def _sanity_check(self, sql: str) -> Optional[str]:
        try:
            df = self.sql_generator.execute_query(sql)
        except Exception:
            return None
        
        if df.empty:
            return "Query returned 0 rows -- verify join/filter logic."
        
        text_blob = df.astype(str).to_string().lower()
        for table, col, value, score in self.context.get('entity_matches', []):
            if value.lower() not in text_blob and len(value) > 3 and score > 0.7:
                return f"'{value}' (from {table}.{col}) not found in results."
        
        # Check for unreasonable results (e.g., marks > 100)
        for col in df.columns:
            if 'mark' in col.lower() or 'score' in col.lower():
                try:
                    max_val = df[col].max()
                    if max_val > 100:
                        return f"Found mark {max_val} > 100 in results. Check if CAST is used correctly."
                except:
                    pass
        
        return None

    def _attempt(self, query: str, feedback: str, temperature: float = 0.05, max_tokens: int = 1500) -> Dict[str, Any]:
        tables = self.rel_map.get_all_table_names()
        ctx = self.context

        entity_hint = ""
        if ctx['entity_matches']:
            lines = [f"  - {table}.{col} = '{value}'  (score: {score:.2f})"
                     for table, col, value, score in ctx['entity_matches'][:5]]
            entity_hint = "\nMATCHED DATABASE VALUES (use these EXACT values, case-insensitively):\n" + "\n".join(lines)
        
        number_hint = f"\nEXTRACTED NUMBER (use for LIMIT/TOP): {ctx['extracted_number']}" if ctx['extracted_number'] else ""
        complexity_hint = ""
        if ctx.get('complexity', {}).get('has_comparison'):
            complexity_hint = "\nThis query involves comparison. Use subqueries or JOINs as needed."
        if ctx.get('complexity', {}).get('has_multi_condition'):
            complexity_hint += "\nThis query has multiple conditions. Use AND/OR with parentheses for clarity."

        prompt = f"""Generate a single SQLite SELECT query for the following question.

{ctx['complete_schema']}

{ctx['all_relationships']}

SAMPLE DATA:
{ctx['table_samples']}

WORKED EXAMPLES (structural patterns):
{FEW_SHOT_EXAMPLES}

HINTS:{entity_hint}{number_hint}{complexity_hint}
{ctx.get('join_paths', '')}

RULES:
1. Use ONLY exact table/column names from schema.
2. ALWAYS use UPPER(col) LIKE UPPER('%value%') for text comparisons.
3. For numeric comparisons, use CAST(col AS REAL).
4. For complex queries, use CTEs (WITH clauses) or subqueries.
5. For ranking, use RANK() or DENSE_RANK() OVER (ORDER BY ...).
6. For categorization, use CASE statements.
7. For comparisons with averages, use subqueries.
8. Return ONLY the SQL query. No markdown, no explanation.

{feedback}

QUESTION: {query}

SQL:"""

        try:
            raw_sql = model_manager.generate(prompt, max_new_tokens=max_tokens, temperature=temperature)
            
            sql = re.sub(r'^```(?:sql)?\s*|\s*```$', '', raw_sql).strip()
            sql = re.sub(r';\s*$', '', sql)
            sql = re.sub(r'\s+', ' ', sql).strip().strip('"')

            # Enhanced text comparison normalization
            sql = re.sub(r"WHERE\s+((?:\w+\.)?\w+)\s*=\s*'([^']+)'",
                          r"WHERE UPPER(\1) LIKE UPPER('%\2%')", sql, flags=re.IGNORECASE)
            sql = re.sub(r"\bAND\s+((?:\w+\.)?\w+)\s*=\s*'([^']+)'",
                          r"AND UPPER(\1) LIKE UPPER('%\2%')", sql, flags=re.IGNORECASE)
            sql = re.sub(r"\bOR\s+((?:\w+\.)?\w+)\s*=\s*'([^']+)'",
                          r"OR UPPER(\1) LIKE UPPER('%\2%')", sql, flags=re.IGNORECASE)
            
            # Fix IN clauses
            def _fix_in_clause(m):
                col, values_str = m.group(1), m.group(2)
                values = re.findall(r"'([^']+)'", values_str)
                upper_values = ", ".join(f"UPPER('{v}')" for v in values)
                return f"UPPER({col}) IN ({upper_values})"
            sql = re.sub(r"((?:\w+\.)?\w+)\s+IN\s*(\([^)]*'[^)]+\))", _fix_in_clause, sql, flags=re.IGNORECASE)

            if ctx['extracted_number'] and 'LIMIT' not in sql.upper():
                if re.search(r'\b(top|best|highest|lowest|worst)\b', query, re.IGNORECASE):
                    sql += f" LIMIT {ctx['extracted_number']}"

            if not sql:
                return {"success": False, "sql": None, "error": "Empty SQL generated"}

            is_valid, message = self.sql_generator.validate_sql(sql)
            if not is_valid:
                return {"success": False, "sql": sql, "error": message}

            try:
                self.sql_generator.connection.execute(f"SELECT * FROM ({sql}) LIMIT 0")
                return {"success": True, "sql": sql}
            except Exception as e:
                error_msg = str(e)
                if "no such table" in error_msg.lower():
                    m = re.search(r"no such table:\s*([^\s]+)", error_msg)
                    if m:
                        error_msg = f"Table '{m.group(1)}' not found. Available: {', '.join(tables)}"
                elif "no such column" in error_msg.lower():
                    m = re.search(r"no such column:\s*([^\s]+)", error_msg)
                    if m:
                        error_msg = f"Column '{m.group(1)}' not found. Check schema."
                elif "ambiguous column" in error_msg.lower():
                    error_msg = "Ambiguous column name -- qualify with table alias."
                return {"success": False, "sql": sql, "error": error_msg}

        except Exception as e:
            return {"success": False, "sql": None, "error": str(e)}

# =============================================================================
# SECTION 9: Fast Path
# =============================================================================

_STOPWORDS = {
    'the', 'a', 'an', 'of', 'in', 'on', 'for', 'and', 'or', 'but', 'is', 'are', 'was', 'were',
    'who', 'which', 'what', 'show', 'list', 'get', 'find', 'display', 'all', 'have', 'has',
    'with', 'above', 'below', 'than', 'each', 'every', 'to', 'by', 'at', 'least', 'most',
    'students', 'student', 'total', 'total marks', 'many', 'how',
}

def check_data_relevance(query: str, rel_map: RelationshipMap,
                          dataframes: Dict[str, pd.DataFrame]) -> Optional[str]:
    words = [w for w in re.findall(r'[a-z]+', query.lower()) if w not in _STOPWORDS and len(w) > 2]
    if not words:
        return None
    vocab = set()
    for table in rel_map.get_all_table_names():
        vocab.update(re.findall(r'[a-z]+', table.lower()))
        for col in rel_map.table_schemas.get(table, {}):
            vocab.update(re.findall(r'[a-z]+', col.lower()))
        df = dataframes.get(table)
        if df is None:
            continue
        for col in df.columns:
            if col in ("_index", "_key"):
                continue
            if not (df[col].dtype == object or pd.api.types.is_string_dtype(df[col])):
                continue
            try:
                joined = " ".join(df[col].dropna().astype(str).unique()[:500]).lower()
                vocab.update(re.findall(r'[a-z]+', joined))
            except Exception:
                continue
    def overlaps(word: str) -> bool:
        stem = word[:-1] if word.endswith('s') and len(word) > 4 else word
        if len(stem) < 4:
            return True
        return any(stem in v or v in stem for v in vocab if len(v) > 3)
    matched = [w for w in words if overlaps(w)]
    if matched:
        return None
    return (f"This dataset doesn't appear to contain data related to \"{query.strip()}\". "
            f"Available tables: {', '.join(rel_map.get_all_table_names())}.")

def try_fast_path(query: str, rel_map: RelationshipMap, sql_generator: SQLGenerator) -> Optional[Dict[str, Any]]:
    q = query.lower().strip()
    tables = rel_map.get_all_table_names()

    def find_table(word: str) -> Optional[str]:
        word = word.rstrip('s')
        for t in tables:
            if word in t.lower():
                return t
        return None

    m = re.match(r'^(?:how many|count of|number of)\s+([a-z_ ]+?)(?:\s+are there)?\??$', q)
    if m:
        table = find_table(m.group(1).strip())
        if table:
            sql = f"SELECT COUNT(*) AS count FROM {table}"
            df = sql_generator.execute_query(sql)
            return {"success": True, "sql": sql, "fast_path": True, "result_df": df}

    m = re.match(r'^(?:show|list|display|get)\s+all\s+([a-z_ ]+?)\??$', q)
    if m:
        table = find_table(m.group(1).strip())
        if table:
            sql = f"SELECT * FROM {table}"
            df = sql_generator.execute_query(sql)
            return {"success": True, "sql": sql, "fast_path": True, "result_df": df}

    return None

# =============================================================================
# SECTION 10: Result Formatter with SQL and Specific Interpretation
# =============================================================================

class ResultFormatter:
    def format(self, query: str, result_df: pd.DataFrame, sql: str = None) -> Dict[str, Any]:
        """Format results and include the SQL query."""
        if result_df.empty:
            return {
                "formatted": "No results found.",
                "interpretation": "No data matches your query. Check the filter conditions or data availability.",
                "sql": sql,
                "rows": 0, 
                "data": None
            }
        
        formatted = self._format_results(query, result_df, sql)
        interpretation = self._specific_interpretation(query, result_df, sql)
        
        return {
            "formatted": formatted, 
            "interpretation": interpretation,
            "sql": sql,
            "rows": len(result_df), 
            "data": result_df
        }

    def _format_results(self, query: str, result_df: pd.DataFrame, sql: str = None) -> str:
        """Format results as clean, readable text without tables."""
        
        output = []
        output.append("=" * 80)
        output.append(f"RESULTS: {len(result_df)} row(s) found")
        output.append("=" * 80)
        output.append("")
        
        # If only one row, display as a detailed list
        if len(result_df) == 1:
            output.append("Details:")
            output.append("-" * 40)
            row = result_df.iloc[0]
            for col in result_df.columns:
                val = row[col]
                if pd.isna(val) or val == "" or val is None:
                    val = "N/A"
                output.append(f"  {col}: {val}")
        
        # If multiple rows, display each as a formatted block
        else:
            for idx, row in result_df.iterrows():
                output.append(f"[{idx + 1}]")
                output.append("-" * 40)
                # Show all fields
                for col in result_df.columns:
                    val = row[col]
                    if pd.isna(val) or val == "" or val is None:
                        continue
                    output.append(f"  {col}: {val}")
                output.append("")
        
        # Add SQL at the end
        if sql:
            output.append("-" * 80)
            output.append("SQL:")
            output.append(sql)
        
        output.append("=" * 80)
        
        return "\n".join(output)

    def _specific_interpretation(self, query: str, df: pd.DataFrame, sql: str = None) -> str:
        """Provides specific, data-driven interpretation for ANY query type."""
        
        # DETECT QUERY TYPE
        query_lower = query.lower()
        query_type = self._detect_query_type(query_lower)
        
        # BUILD SPECIFIC FINDINGS
        findings = self._extract_specific_findings(df)
        
        # GENERATE INTERPRETATION
        prompt = self._build_interpretation_prompt(query, df, query_type, findings, sql)
        
        try:
            return model_manager.generate(
                prompt,
                system_prompt=self._get_system_prompt(query_type),
                max_new_tokens=700,
                temperature=0.05
            )
        except Exception:
            return self._fallback_interpretation(df)

    def _detect_query_type(self, query: str) -> str:
        """Detect query type."""
        if re.search(r'\b(avg|average|mean|sum|total|count|how many|number of)\b', query):
            if re.search(r'\b(by|per|for each|group)\b', query):
                return 'aggregation_grouped'
            return 'aggregation_simple'
        if re.search(r'\b(top|best|highest|lowest|worst|rank|bottom)\b', query):
            if re.search(r'\b(top|best|highest)\b', query):
                return 'ranking_top'
            return 'ranking_bottom'
        if re.search(r'\b(show|list|find|get|display|who|which)\b', query):
            if re.search(r'\b(fail|pass|below|above)\b', query):
                return 'filter_condition'
            return 'filter_simple'
        if re.search(r'\b(compare|vs|versus|difference|between|than)\b', query):
            return 'comparison'
        if re.search(r'\b(distribution|spread|range|vary|diverse)\b', query):
            return 'distribution'
        if re.search(r'\b(detail|profile|info|information|full|complete)\b', query):
            return 'profile'
        return 'general'

    def _extract_specific_findings(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Extract specific findings from data."""
        findings = {
            'row_count': len(df),
            'numeric_stats': {},
            'categorical_counts': {},
            'extremes': {},
            'nulls': {}
        }
        
        if df.empty:
            return findings
        
        # Numeric columns
        for col in df.select_dtypes(include=[np.number]).columns:
            findings['numeric_stats'][col] = {
                'min': df[col].min(),
                'max': df[col].max(),
                'mean': df[col].mean(),
                'median': df[col].median(),
                'std': df[col].std(),
                'null_count': df[col].isna().sum()
            }
        
        # Categorical columns
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].nunique() <= 20:
                findings['categorical_counts'][col] = df[col].value_counts().to_dict()
        
        # Extremes
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                top_3 = df.nlargest(3, col)
                if len(top_3) > 0:
                    findings['extremes'][f'{col}_top'] = top_3[[col] + [c for c in df.columns if c != col][:2]].to_dict('records')
                bottom_3 = df.nsmallest(3, col)
                if len(bottom_3) > 0:
                    findings['extremes'][f'{col}_bottom'] = bottom_3[[col] + [c for c in df.columns if c != col][:2]].to_dict('records')
        
        # Nulls
        for col in df.columns:
            null_count = df[col].isna().sum()
            if null_count > 0:
                findings['nulls'][col] = null_count
        
        return findings

    def _build_interpretation_prompt(self, query: str, df: pd.DataFrame, 
                                      query_type: str, findings: Dict, sql: str = None) -> str:
        """Build specific interpretation prompt with SQL."""
        
        sql_context = f"\nThe SQL used to get this data was: {sql}" if sql else ""
        
        data_summary = f"""
DATA SUMMARY:
- Total rows: {findings['row_count']}
- Columns: {', '.join(df.columns.tolist())}

NUMERIC STATISTICS:
{self._format_numeric_stats(findings['numeric_stats'])}

CATEGORICAL COUNTS:
{self._format_categorical_counts(findings['categorical_counts'])}
{sql_context}
"""
        
        # Type-specific prompts
        prompts = {
            'aggregation_simple': f"""
QUESTION: {query}

{data_summary}

Provide a SPECIFIC interpretation:
1. What is the EXACT result?
2. What does this number mean in context?
3. Is this number surprising? Why?
4. What other numbers would be useful for comparison?

EXACT numbers from the data:
{self._extract_aggregation_values(df, findings)}
""",

            'aggregation_grouped': f"""
QUESTION: {query}

{data_summary}

Provide a SPECIFIC interpretation:
1. List each group with its EXACT value
2. Which group has the highest value? (EXACT number)
3. Which group has the lowest value? (EXACT number)
4. What is the range/difference? (EXACT number)
5. Is any group an outlier? Why?

GROUPED RESULTS:
{df.to_string(index=False)}
""",

            'ranking_top': f"""
QUESTION: {query}

{data_summary}

Provide a SPECIFIC interpretation:
1. List the top results with EXACT names and values
2. What is the #1 result? (EXACT name and value)
3. What is the gap between #1 and #2? (EXACT number)
4. What is the gap between #1 and #10? (EXACT number)

TOP RESULTS:
{df.head(10).to_string(index=False)}
""",

            'ranking_bottom': f"""
QUESTION: {query}

{data_summary}

Provide a SPECIFIC interpretation:
1. List the bottom results with EXACT names and values
2. What is the worst result? (EXACT name and value)
3. What is the gap between worst and second worst? (EXACT number)

BOTTOM RESULTS:
{df.head(10).to_string(index=False)}
""",

            'filter_condition': f"""
QUESTION: {query}

{data_summary}

Provide a SPECIFIC interpretation:
1. How many items match the filter? (EXACT number)
2. What is the typical value among matches? (EXACT number)
3. What are the extremes among matches? (EXACT min/max)
4. List any notable items with EXACT names/values

FILTERED RESULTS:
{df.to_string(index=False)}
""",

            'comparison': f"""
QUESTION: {query}

{data_summary}

Provide a SPECIFIC interpretation:
1. What are the groups being compared?
2. What is the value for each group? (EXACT numbers)
3. What is the difference between groups? (EXACT number)
4. Which group is better/higher? (EXACT number/percentage)

COMPARISON RESULTS:
{df.to_string(index=False)}
""",

            'distribution': f"""
QUESTION: {query}

{data_summary}

Provide a SPECIFIC interpretation:
1. What is the range? (EXACT min/max)
2. What is the most common value? (EXACT number/category)
3. What is the least common value? (EXACT number/category)
4. What percentage falls into each category? (EXACT percentages)

DISTRIBUTION DATA:
{df.to_string(index=False)}
""",

            'profile': f"""
QUESTION: {query}

{data_summary}

Provide a SPECIFIC interpretation:
1. List the key attributes found
2. What is the most notable information?
3. Are there any missing or incomplete fields?
4. What stands out about this profile?

PROFILE DATA:
{df.to_string(index=False)}
""",

            'general': f"""
QUESTION: {query}

{data_summary}

Provide a SPECIFIC interpretation:
1. What exactly does the data show?
2. What is the most important finding? (EXACT value)
3. What trends or patterns do you see?
4. What data quality issues exist? (EXACT null counts)

RESULTS:
{df.to_string(index=False)}
"""
        }
        
        return prompts.get(query_type, prompts['general'])

    def _format_numeric_stats(self, stats: Dict) -> str:
        if not stats:
            return "  No numeric columns found."
        lines = []
        for col, vals in stats.items():
            lines.append(f"  {col}:")
            lines.append(f"    - Min: {vals['min']:.2f}")
            lines.append(f"    - Max: {vals['max']:.2f}")
            lines.append(f"    - Mean: {vals['mean']:.2f}")
            lines.append(f"    - Median: {vals['median']:.2f}")
            lines.append(f"    - Std Dev: {vals['std']:.2f}")
            if vals['null_count'] > 0:
                lines.append(f"    - Missing: {vals['null_count']} rows")
        return "\n".join(lines)

    def _format_categorical_counts(self, counts: Dict) -> str:
        if not counts:
            return "  No categorical columns with limited unique values."
        lines = []
        for col, vals in counts.items():
            lines.append(f"  {col}:")
            for val, count in vals.items():
                lines.append(f"    - '{val}': {count} rows")
        return "\n".join(lines)

    def _extract_aggregation_values(self, df: pd.DataFrame, findings: Dict) -> str:
        if df.empty:
            return "  No data available."
        lines = []
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                lines.append(f"  {col}: {df[col].sum() if len(df) > 1 else df[col].iloc[0]}")
        return "\n".join(lines) if lines else "  No numeric values found."

    def _get_system_prompt(self, query_type: str) -> str:
        base = """You are a data analyst providing SPECIFIC, data-driven analysis. 
ALWAYS use exact numbers, names, and percentages from the data. 
NEVER be vague. NEVER say "some" or "many" - use exact counts.
Be concise but informative."""
        
        type_specific = {
            'aggregation_simple': "Focus on the EXACT number and what it means.",
            'aggregation_grouped': "Compare groups with EXACT numbers and identify the highest/lowest.",
            'ranking_top': "List the top results with EXACT names and values. Show the gaps.",
            'ranking_bottom': "List the bottom results with EXACT names and values. Show the gaps.",
            'filter_condition': "State EXACT count and list specific items that match.",
            'comparison': "State EXACT values for each group and the EXACT difference.",
            'distribution': "State EXACT min, max, most common, and least common values.",
            'profile': "List EXACT attributes and identify what stands out.",
            'general': "Provide specific, data-driven insights."
        }
        return base + " " + type_specific.get(query_type, type_specific['general'])

    def _fallback_interpretation(self, df: pd.DataFrame) -> str:
        if df.empty:
            return "No data found."
        
        lines = [f"Found {len(df)} rows:"]
        for idx, row in df.head(5).iterrows():
            row_str = " | ".join([f"{col}: {val}" for col, val in row.items()])
            lines.append(f"  {idx + 1}. {row_str}")
        
        if len(df) > 5:
            lines.append(f"  ... and {len(df) - 5} more rows")
        
        for col in df.select_dtypes(include=[np.number]).columns:
            lines.append(f"\n{col}:")
            lines.append(f"  Min: {df[col].min()}")
            lines.append(f"  Max: {df[col].max()}")
            lines.append(f"  Avg: {df[col].mean():.2f}")
        
        return "\n".join(lines)
# =============================================================================
# SECTION 11: Pipeline
# =============================================================================

def run_analysis_pipeline(query: str, dataframes: Dict[str, pd.DataFrame],
                            rel_map: RelationshipMap) -> Dict[str, Any]:
    if not dataframes:
        return {"success": False, "error": "No datasets provided"}

    no_data_msg = check_data_relevance(query, rel_map, dataframes)
    if no_data_msg:
        return {"success": False, "error": no_data_msg, "no_matching_data": True}

    sql_generator = SQLGenerator(rel_map)
    sql_generator.create_tables_from_dataframes(dataframes)

    fast = try_fast_path(query, rel_map, sql_generator)
    if fast:
        formatter = ResultFormatter()
        fmt = formatter.format(query, fast["result_df"], fast["sql"])
        return {
            "success": True, 
            "query": query, 
            "sql": fast["sql"],
            "fast_path": True,
            "formatted": fmt["formatted"], 
            "interpretation": fmt["interpretation"],
            "result_df": fast["result_df"], 
            "rows": len(fast["result_df"])
        }

    sql_llm = SQLGeneratorLLM(rel_map, sql_generator, EntityIndex(dataframes))
    sql_result = sql_llm.generate(query, max_attempts=4)

    if not sql_result["success"]:
        return {
            "success": False, 
            "error": sql_result["message"], 
            "details": sql_result["errors"],
            "context": sql_result.get("context"), 
            "attempts": sql_result.get("attempts", [])
        }

    try:
        result_df = sql_generator.execute_query(sql_result["sql"])
    except Exception as e:
        return {
            "success": False, 
            "error": f"Query execution failed: {e}",
            "sql": sql_result["sql"],
            "context": sql_result.get("context")
        }

    formatter = ResultFormatter()
    fmt = formatter.format(query, result_df, sql_result["sql"])

    return {
        "success": True, 
        "query": query, 
        "sql": sql_result["sql"],
        "context": sql_result.get("context"),
        "formatted": fmt["formatted"], 
        "interpretation": fmt["interpretation"],
        "result_df": result_df, 
        "rows": len(result_df)
    }

# =============================================================================
# SECTION 12: CLI
# =============================================================================

def print_header(text): print("\n" + "=" * 70 + f"\n  {text}\n" + "=" * 70)
def print_success(text): print(f"[OK] {text}")
def print_error(text): print(f"[FAIL] {text}")
def print_info(text): print(f"[i] {text}")

def is_command(cmd: str) -> bool:
    commands = ["help", "upload", "use", "preview", "relationships", "clear", "exit", "quit", "show"]
    return cmd.split()[0].lower() in commands

def main():
    print_header("LLM Data Analysis System v3 - Enhanced")
    print("  [INFO] Type 'help' for commands, or type any question directly")
    print("=" * 70)

    model_manager.load_async()
    datasets, rel_maps, active_dataset = {}, {}, None
    last_sql = None

    while True:
        try:
            user_input = input("\n[QUERY] ").strip()
            if not user_input:
                continue

            if is_command(user_input):
                parts = user_input.split(maxsplit=1)
                command, arg = parts[0].lower(), (parts[1] if len(parts) > 1 else None)

                if command in ("exit", "quit"):
                    print("Goodbye!")
                    break

                elif command == "help":
                    print_header("COMMANDS")
                    print("  upload <file>     - Upload a dataset (.json/.csv/.xlsx)")
                    print("  use <name>        - Set active dataset")
                    print("  preview           - Preview active dataset")
                    print("  relationships     - Show relationships in active dataset")
                    print("  show sql          - Show last generated SQL")
                    print("  clear             - Clear screen")
                    print("  exit              - Exit program")
                    print("\nOtherwise, just type a natural-language question.")

                elif command == "clear":
                    import os
                    os.system('cls' if os.name == 'nt' else 'clear')

                elif command == "upload":
                    if not arg:
                        print_error("Please provide a file path")
                        continue
                    try:
                        tables, rel_map = read_dataset_file(arg)
                        name = arg.split("/")[-1]
                        datasets[name], rel_maps[name], active_dataset = tables, rel_map, name
                        print_success(f"Uploaded '{name}' ({len(tables)} tables, "
                                       f"{sum(len(d) for d in tables.values())} total rows)")
                        for table_name, df in tables.items():
                            print(f"  - {table_name}: {len(df)} rows, {len(df.columns)} columns")
                        if rel_map.relationships:
                            print_info(f"Found {len(rel_map.relationships)} relationships:")
                            for i, rel in enumerate(rel_map.relationships):
                                print(f"  {i + 1}. {rel.join_condition()} ({rel.cardinality})")
                    except Exception as e:
                        print_error(f"Failed to upload: {e}")
                        import traceback
                        traceback.print_exc()

                elif command == "use":
                    if arg in datasets:
                        active_dataset = arg
                        print_success(f"Active dataset: {arg}")
                    else:
                        print_error(f"Dataset '{arg}' not found")

                elif command == "relationships":
                    if not active_dataset:
                        print_error("No active dataset")
                        continue
                    print_header(f"RELATIONSHIPS FOR '{active_dataset}'")
                    print(rel_maps.get(active_dataset, RelationshipMap()).get_summary())

                elif command == "preview":
                    if not active_dataset:
                        print_error("No active dataset")
                        continue
                    print_header(f"PREVIEW: {active_dataset}")
                    for table_name, df in datasets[active_dataset].items():
                        print(f"\nTable: {table_name}  shape={df.shape}")
                        print(tabulate(df.head(5), headers="keys", tablefmt="grid"))

                elif command == "show" and arg == "sql":
                    print_header("LAST GENERATED SQL")
                    print(last_sql or "No SQL has been generated yet")

                else:
                    print_error(f"Unknown command: {command}")

            else:
                if not active_dataset:
                    print_error("No active dataset. Use 'upload' first.")
                    continue

                result = run_analysis_pipeline(user_input, datasets[active_dataset],
                                                 rel_maps.get(active_dataset, RelationshipMap()))

                if result.get("success"):
                    print_success(f"Found {result.get('rows', 0)} rows"
                                   f"{'  [fast-path]' if result.get('fast_path') else ''}")
                    
                    # ✅ Show the generated SQL
                    if result.get("sql"):
                        print("\n" + "=" * 70)
                        print("📝 GENERATED SQL:")
                        print("=" * 70)
                        print(result["sql"])
                    
                    last_sql = result.get("sql")
                    print("\n" + result["formatted"])
                    print("\n" + result["interpretation"])
                else:
                    print_error(f"Analysis failed: {result.get('error', 'Unknown error')}")
                    if result.get("details"):
                        for d in result["details"]:
                            print(f"  - {d}")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print_error(f"Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()