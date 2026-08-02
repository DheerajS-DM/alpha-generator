import os
import json
import logging
import time
import ast
import re
from typing import Dict, List
from groq import Groq
from google import genai
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Forbidden methods that Polars doesn't support
FORBIDDEN_METHODS = [
    '.apply(',
    '.map(',
    '.transform(',
    '.applymap(',
    '.assign(',
]

def contains_forbidden_methods(code: str) -> tuple[bool, str]:
    """Check if code contains forbidden Polars methods"""
    for method in FORBIDDEN_METHODS:
        if method in code:
            return True, method
    # Check for common anti-patterns
    if '.over(' in code and code.count('.over(') > 1:
        return True, 'nested .over() calls'
    return False, None
class MultiAPITranspiler:
    def __init__(self, operator_path: str = "operatorRAW.json"):
        # Load the "Source of Truth"
        with open(operator_path, 'r') as f:
            self.raw_operators = json.load(f)
            
        # Extract relevant technical metadata for the LLM
        self.doc_context = self._build_doc_context()
        
        # API Clients initialization (unchanged)
        self.groq_clients = [Groq(api_key=os.environ.get(f"GROQ_API_KEY_{i}")) for i in [1, 2, 3] if os.environ.get(f"GROQ_API_KEY_{i}")]
        self.gemini_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
        self.gemini_model_name = "gemini-1.5-pro"
        self.current_groq_idx = 0

    def _build_doc_context(self) -> str:
        """Converts JSON operators into a technical manual for the LLM."""
        docs = []
        for op in self.raw_operators:
            if op.get("level") == "ALL": # Only include standard functions
                docs.append(f"Function: {op['name']}\nParams: {op.get('definition', 'N/A')}\nDescription: {op.get('description', '')}")
        return "\n---\n".join(docs)

    def generate_prompt(self, formulas: List[str]) -> str:
        return f"""
        You are a compiler transpiling WorldQuant Brain formulas into Polars pl.Expr code.
        
        ### TECHNICAL DOCUMENTATION:
        Use the following function definitions to understand the intent and parameters of the formulas:
        {self.doc_context}
        
        ### CRITICAL POLARS SYNTAX RULES - VALID METHODS ONLY:
        Valid methods on Expr objects: .rolling_mean(), .rolling_sum(), .rank(), .mean(), .std(), .count(), .max(), .min(), .sum(), .shift(), .diff()
        Valid functions in pl namespace: pl.col(), pl.lit(), pl.when(), pl.otherwise(), pl.rolling_corr(), pl.rolling_cov()
        
        ### FORBIDDEN - NEVER USE THESE:
        - NEVER use .apply() - Polars Expr objects do NOT have an .apply() method
        - NEVER use .map() - use .rolling_mean() or other specific methods instead
        - NEVER use .transform() - not a Polars method
        - NEVER use nested .over().over() - only ONE .over() per expression
        - NEVER use .rolling_corr() or .rolling_cov() as methods on Expr - use pl.rolling_corr() or pl.rolling_cov() from namespace
        
        ### TRANSLATION RULES:
        - Use `pl.col('name')` for fields
        - Time-series: `ts_mean(x, 20)` -> `pl.col('x').rolling_mean(window_size=20)`
        - Rank: `rank(x)` -> `pl.col('x').rank() / pl.col('x').count()`
        - Normalize: `normalize(x)` -> `(pl.col('x') - pl.col('x').mean()) / pl.col('x').std()`
        - Lag: `ts_delay(x, 5)` -> `pl.col('x').shift(5)`
        - Delta: `ts_delta(x, 5)` -> `pl.col('x') - pl.col('x').shift(5)`
        - Arithmetic: `add(x, y)` -> `pl.col('x') + pl.col('y')`
        - If unsure about a function, substitute with a simple numeric operation
        
        ### OUTPUT REQUIREMENTS:
        - Generate ONLY valid Python expressions that can be evaluated with eval()
        - Each formula's output must be a single pl.Expr object
        - No comments, no multi-line code, just the expression
        - If a formula is too complex, simplify it to basic arithmetic or ts_mean only
        
        Translate these formulas into a JSON object where the key is the formula and the value is the Polars code.
        Formulas: {json.dumps(formulas)}
        """
    
    # ... rest of transpile_batch logic remains the same ...

    def transpile_batch(self, formulas: List[str]) -> Dict[str, str]:
        prompt = self.generate_prompt(formulas)
        
        # --- Attempt 1: Groq Key Rotation ---
        if self.groq_clients:
            attempts = 0
            max_attempts = len(self.groq_clients)
            
            while attempts < max_attempts:
                client = self.groq_clients[self.current_groq_idx]
                try:
                    logger.info(f"Attempting batch via Groq (Key {self.current_groq_idx + 1}/{max_attempts})...")
                    response = client.chat.completions.create(
                        messages=[{"role": "user", "content": prompt}],
                        # FIX: Updated to Groq's current supported Llama 3.3 model
                        model="llama-3.3-70b-versatile", 
                        temperature=0.1,
                        response_format={"type": "json_object"}
                    )
                    raw_json = response.choices[0].message.content
                    parsed_json = json.loads(raw_json)
                    
                    # Validate that all values are syntactically valid Polars expressions
                    valid_json = {}
                    for formula, polars_code in parsed_json.items():
                        # Check for syntax errors
                        try:
                            ast.parse(polars_code, mode='eval')
                        except SyntaxError as se:
                            logger.warning(f"❌ Syntax error in formula: {formula}")
                            logger.warning(f"   Code: {polars_code[:80]}...")
                            logger.warning(f"   Error: {se}")
                            continue
                        
                        # Check for forbidden methods
                        is_forbidden, forbidden_item = contains_forbidden_methods(polars_code)
                        if is_forbidden:
                            logger.warning(f"❌ Forbidden method in formula: {formula}")
                            logger.warning(f"   Code: {polars_code[:80]}...")
                            logger.warning(f"   Forbidden: {forbidden_item}")
                            continue
                        
                        valid_json[formula] = polars_code
                        logger.debug(f"✓ Valid formula: {formula[:50]}... -> {polars_code[:60]}...")
                    
                    if valid_json:
                        return valid_json
                    else:
                        logger.warning(f"No valid formulas after validation. All {len(parsed_json)} were rejected.")
                    
                    return valid_json
                    
                except Exception as e:
                    if "429" in str(e) or "rate_limit" in str(e).lower():
                        logger.warning(f"Groq Key {self.current_groq_idx + 1} rate limited. Rotating to next key...")
                        self.current_groq_idx = (self.current_groq_idx + 1) % len(self.groq_clients)
                        attempts += 1
                    else:
                        logger.error(f"Groq failed with non-rate-limit error: {e}. Aborting Groq loop.")
                        break

        # --- Attempt 2: Gemini (Ultimate Failover) ---
        try:
            logger.info(f"Groq exhausted/failed. Routing batch to {self.gemini_model_name}...")
            response = self.gemini_client.models.generate_content(
                model="models/" + self.gemini_model_name,
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.1
                )
            )
            raw_text = response.text
            
            if raw_text.startswith("```"):
                raw_text = raw_text.strip("`").removeprefix("json").strip()
                
            parsed_json = json.loads(raw_text)
            
            # Validate that all values are syntactically valid Polars expressions
            valid_json = {}
            for formula, polars_code in parsed_json.items():
                # Check for syntax errors
                try:
                    ast.parse(polars_code, mode='eval')
                except SyntaxError as se:
                    logger.warning(f"❌ Gemini syntax error in formula: {formula}")
                    logger.warning(f"   Code: {polars_code[:80]}...")
                    logger.warning(f"   Error: {se}")
                    continue
                
                # Check for forbidden methods
                is_forbidden, forbidden_item = contains_forbidden_methods(polars_code)
                if is_forbidden:
                    logger.warning(f"❌ Gemini forbidden method in formula: {formula}")
                    logger.warning(f"   Code: {polars_code[:80]}...")
                    logger.warning(f"   Forbidden: {forbidden_item}")
                    continue
                
                valid_json[formula] = polars_code
                logger.debug(f"✓ Gemini valid formula: {formula[:50]}... -> {polars_code[:60]}...")
            
            if valid_json:
                return valid_json
            else:
                logger.warning(f"Gemini: No valid formulas after validation. All {len(parsed_json)} were rejected.")
            
            return valid_json
            
        except json.JSONDecodeError:
            logger.error(f"Gemini returned invalid JSON. Raw output: {raw_text[:100]}...")
            return {}
        except Exception as e:
            logger.error(f"Gemini failover failed: {e}")
            time.sleep(15)
            return {}