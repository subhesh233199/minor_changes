import os
import re
import json
import runpy
import base64
import sqlite3
import hashlib
import time
from typing import List, Dict, Tuple, Any, Union
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from crewai import Agent, Task, Crew, Process, LLM
from langchain_openai import AzureChatOpenAI
import ssl
import warnings
import shutil
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, validator
import logging
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tenacity import retry, stop_after_attempt, wait_fixed
from copy import deepcopy
import pdfplumber
from crewai import Agent, Task, Crew, Process  # Ensure imports in function scope for clarity
from shared_state import shared_state
from app_logging import logger
from models import (FolderPathRequest
                    ,AnalysisResponse 
                    , MetricItem)
from utils import (convert_windows_path
                   ,get_base64_image
                   ,get_pdf_files_from_folder
                   ,extract_hyperlinks_from_pdf
                   ,enhance_report_markdown
                   ,evaluate_with_llm_judge)
from cache_utils import (init_cache_db
                         ,hash_string
                         ,hash_pdf_contents
                         ,get_cached_report
                         ,store_cached_report
                         ,cleanup_old_cache)



# Disable SSL verification
ssl._create_default_https_context = ssl._create_unverified_context

# Suppress warnings
warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()

#Constants
RELEASE_SCOPE_METRICS = [
    "Target Customers",
    "Release Epics",
    "Release PIRs",
    "SFDC Defects Fixed"
]
RELEASE_SCOPE_COLUMNS = [
    "Data Source",
    "Description",
    "ATLs Fixed",
    "BTLs Fixed",
    "Total Fixed",
    "Comments"
]
CRITICAL_METRICS = [
    "Delivery Against Requirements",
    "System / Solution Test Metrics",
    "System / Solution Test Coverage",
    "System / Solution Test Pass Rate",
    "Security Test Metrics",
    "Performance / Load Test Metrics"
]
CRITICAL_COLUMNS = [
    "Functional Group",
    "Release Criteria",
    "Metrics (WST {Version})",
    "Status"
]
HEALTH_TRENDS_METRICS = [
    "Unit Test Coverage",
    "Automation Test Coverage"
]
HEALTH_TRENDS_COLUMNS = [
    "Health Metrics",
    "Release Criteria",
    "Previous Release Metrics",
    "Current Release Metrics (WST {Version})",
    "Status",
    "Summary / Take-Away / Comments"
]



table_indices = {
    "release_approval":(1, 1),    # (page_number, table_index)
    "release_scope": (1, 2),
    "key_stakeholders":(1, 3),         
    "critical_metrics": (2, 2),
    "health_trends": (2, 8)
}

# Initialize Azure OpenAI
llm = LLM(
    model=f"azure/{os.getenv('DEPLOYMENT_NAME')}",
    api_version=os.getenv("AZURE_API_VERSION"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    base_url=os.getenv("AZURE_OPENAI_ENDPOINT"),
    temperature=0.1,
    top_p=0.95,
)

def normalize_metrics(metrics):
    """
    Restructure LLM output into a {metric: {version: {"ATLs Fixed": x, "BTLs Fixed": y}}} format.
    Applies to both release_scope and critical_metrics.
    """
    import re

    def safe_float(val):
        try:
            if isinstance(val, str):
                # extract the first number, ignore units/words
                digits = "".join([c for c in val if c.isdigit() or c == "."])
                return float(digits) if digits else None
            return float(val)
        except Exception:
            return None

    # For Release Scope
    if "release_scope" in metrics:
        orig = metrics["release_scope"]
        result = {}
        for metric, version_dict in orig.items():
            # If the LLM output is in per-version, per-metric list format, handle that:
            if isinstance(version_dict, dict):
                # new format: metric: {version: {"ATLs Fixed": .., "BTLs Fixed": ..}}
                result[metric] = {}
                for version, vals in version_dict.items():
                    atls = safe_float(vals.get("ATLs Fixed"))
                    btls = safe_float(vals.get("BTLs Fixed"))
                    result[metric][version] = {"ATLs Fixed": atls, "BTLs Fixed": btls}
            # fallback: legacy per-version format
            else:
                continue
        metrics["release_scope"] = result

    # For Critical Metrics
    if "critical_metrics" in metrics:
        orig = metrics["critical_metrics"]
        result = {}
        for metric, version_dict in orig.items():
            result[metric] = {}
            for version, vals in version_dict.items():
                atls = safe_float(vals.get("ATLs Fixed"))
                btls = safe_float(vals.get("BTLs Fixed"))
                result[metric][version] = {"ATLs Fixed": atls, "BTLs Fixed": btls}
        metrics["critical_metrics"] = result

    return metrics

def extract_tables_text_for_version(pdf_path, version, indices_dict):
    """
    Extracts named tables as strings from a PDF, prepends version label,
    and returns a single combined string in the expected order/format.
    """

    all_tables = []
    with pdfplumber.open(pdf_path) as pdf:
        for pagenum, page in enumerate(pdf.pages, 1):
            tables = page.extract_tables()
            for t_idx, table in enumerate(tables, 1):
                all_tables.append({
                    "page": pagenum,
                    "table_index": t_idx,
                    "data": table
                })

    def get_table(page, table_idx):
        for t in all_tables:
            if t["page"] == page and t["table_index"] == table_idx:
                return t["data"]
        return None

    final_text = f"Version {version}\n"
    for label, (page, idx) in indices_dict.items():
        table = get_table(page, idx)
        final_text += f"\n{label.replace('_', ' ').title()}\n"
        if table:
            import pandas as pd
            df = pd.DataFrame(table[1:], columns=table[0])
            final_text += df.to_string(index=False) + "\n"
        else:
            final_text += "No data found.\n"

    return final_text.strip()


def setup_crew_wst(extracted_text: str, versions: list, llm=llm):
    """
    Sets up CrewAI agents and tasks for the WST (Scheduling/Timekeeping) product.
    Returns: data_crew, report_crew, viz_crew, brief_summary_crew
    """

    # 1. Data Crew Agent & Task
    data_agent = Agent(
        role="Data Architect",
        goal="Structure raw WST release data into VALID JSON format",
        backstory=(
            "Expert in transforming unstructured release readiness report data "
            "for Scheduling & Timekeeping (WST) into clean JSON, with strict adherence to metrics/columns."
        ),
        llm=llm,
        verbose=True,
        memory=True,
    )

    DATA_CREW_PROMPT = f"""
Given the combined text for multiple versions...
Your tasks:
- For each table, output a JSON object with metric names as keys, each containing a dict mapping version to values for ATLs Fixed and BTLs Fixed.
- For example:
"release_scope": {{
  "Release Epics": {{
    "45.1.15.0": {{"ATLs Fixed": X, "BTLs Fixed": Y}},
    "45.1.16.0": {{"ATLs Fixed": ..., "BTLs Fixed": ...}},
    ...
  }},
  ...
}}
...
Return only JSON, no commentary.
"""

    data_task = Task(
        description=DATA_CREW_PROMPT,
        agent=data_agent,
        output_key="structured_json",
        assign_output=lambda output: setattr(shared_state, "metrics", output),
        context=[],
        expected_output="A valid JSON object with top-level keys 'release_scope', 'critical_metrics', and 'health_trends', each containing data for all versions and all specified metrics/columns."
    )
    data_crew = Crew(name="data_crew", agents=[data_agent], tasks=[data_task])

    # 2. Report Crew Agent & Task (with enhanced Overview prompt)
    report_agent = Agent(
        role="Technical Writer",
        goal="Generate a detailed software release metrics markdown report for WST",
        backstory="Expert in software release reporting, follows strict structure for WST.",
        llm=llm,
        verbose=True,
        memory=True,
    )

    REPORT_CREW_PROMPT = """
# Software Metrics Report

## Overview
Write a detailed executive overview (at least 8–10 sentences) for a business/leadership audience:
- State the overall purpose of the analyzed releases and the versions compared.
- Highlight major trends and themes visible across all three tables (“Release Scope”, “Critical Metrics”, “Health Trends”), including cross-version comparisons.
- Call out significant improvements, regressions, or consistent patterns across versions, referencing specific metrics and values (e.g., increases in ATLs Fixed, improvements in Test Coverage, changes in pass rates).
- Identify any ongoing risks or problem areas that persist across multiple releases, citing numbers or observed trends.
- Mention any new issues, resolved blockers, or notable “firsts” for this product’s release cycle.
- Summarize the general health and readiness of the software based on all available data.
- Compare to prior release cycles or industry benchmarks if relevant.
- End with a strong statement about the release outlook or readiness for production, backed by specific evidence from the metrics.
Be specific, reference numbers/percentages, and synthesize data across tables. Use formal business English.

---

## Metrics Summary
[Generate, for each metric below, a separate markdown table with all versions as rows, columns as specified, and trend calculation as described. Present metrics/tables in the exact order given below, even if data is missing. Show N/A for missing.]

(REPEAT: All markdown tables must match column order, use trend % and arrow, and include all required metrics for each table.)

- Release Scope Metrics:
  Metrics: Target Customers, Release Epics, Release PIRs, SFDC Defects Fixed
  Columns: Data Source, Description, ATLs Fixed, BTLs Fixed, Total Fixed, Comments, Trend

- Critical Metrics:
  Metrics: Delivery Against Requirements, System / Solution Test Metrics, System / Solution Test Coverage, System / Solution Test Pass Rate, Security Test Metrics, Performance / Load Test Metrics
  Columns: Functional Group, Release Criteria, Metrics (WST {Version}), Status, Trend

- Health Trend Table:
  Metrics: Unit Test Coverage, Automation Test Coverage
  Columns: Health Metrics, Release Criteria, Previous Release Metrics, Current Release Metrics (WST {Version}), Status, Summary / Take-Away / Comments, Trend

## Key Findings
[Summarize most important findings per metric group.]

---

## Recommendations
[Write actionable recommendations based on trends and status.]

- For any missing data, show "N/A"
- Use arrows: ↑ for increase, ↓ for decrease, → for no change (w/ %)
- If no prior version, show "N/A" in Trend.
"""

    report_task = Task(
        description=REPORT_CREW_PROMPT,
        agent=report_agent,
        output_key="markdown_report",
        context=[],
        expected_output="A professional markdown report summarizing all WST release metrics, with tables, key findings, and recommendations as per the prompt."
    )
    report_crew = Crew(name="report_crew", agents=[report_agent], tasks=[report_task])

    # 3. Visualization Crew Agent & Task
    viz_agent = Agent(
        role="Data Visualization Specialist",
        goal="Generate Python matplotlib scripts for WST release metrics",
        backstory="Expert in creating charts for software release reporting. Uses pandas/matplotlib.",
        llm=llm,
        verbose=True,
        memory=True,
    )

    VIZ_CREW_PROMPT = """
Given structured JSON with three tables ("Release Scope", "Critical Metrics", "Health Trends") per version for WST, generate Python matplotlib scripts for each metric as follows:

- Release Scope: For each metric (Target Customers, Release Epics, Release PIRs, SFDC Defects Fixed), plot a line chart of "ATLs Fixed" vs version.
- Critical Metrics: For each metric, plot line chart of ATLs (from "Metrics (WST {Version})") vs version.
- Health Trends: For each metric, plot line chart of "Current Release Metrics (WST {Version})" vs version.
- For each table, create a grouped bar chart comparing all relevant metrics' values across versions.
- Annotate >20% increases/decreases with arrows.
- Use matplotlib and pandas, no seaborn, no custom colors.
- Each chart/code block is independent and executable.
- Add markdown header above each code block with chart description.

If data is missing, handle gracefully.
"""

    viz_task = Task(
        description=VIZ_CREW_PROMPT,
        agent=viz_agent,
        output_key="viz_code",
        context=[],
        expected_output="A set of Python matplotlib scripts (as code blocks) to visualize all WST release metrics trends and comparisons."
    )
    viz_crew = Crew(name="viz_crew", agents=[viz_agent], tasks=[viz_task])

    # 4. Brief Summary Crew (corrected: *real* brief summary for busy leaders)
    brief_summary_agent = Agent(
        role="Brief Summary Specialist",
        goal="Write a short, high-level summary for WST business leaders.",
        backstory="Summarizes the main takeaway for busy executives. Just the most important point, no technical or detailed analysis.",
        llm=llm,
        verbose=True,
        memory=True,
    )
    BRIEF_SUMMARY_PROMPT = """
Write a **concise business summary** of the WST release metrics, for senior management who only have 2 minutes to read.

- Limit your summary to 2–4 sentences, maximum 100 words.
- Cover only the most important trend, risk, or positive highlight from the current and previous releases.
- Avoid technical jargon, tables, or detailed breakdowns.
- If there are any major risks or improvements, mention those in simple language.
- End with a strong, clear statement about overall release health or business impact.

Respond in plain, high-level business English.
"""

    brief_summary_task = Task(
        description=BRIEF_SUMMARY_PROMPT,
        agent=brief_summary_agent,
        output_key="brief_summary",
        context=[],  # Optionally, you could pass the main report as context if you want
        expected_output="A short, high-level summary (2–4 sentences, max 100 words) focused on the biggest insight for business leaders."
    )
    brief_summary_crew = Crew(name="brief_summary_crew", agents=[brief_summary_agent], tasks=[brief_summary_task])

    return data_crew, report_crew, viz_crew, brief_summary_crew





def run_fallback_visualization(metrics):
    """
    Generates line charts for ATLs and BTLs for each metric in Release Scope and Critical Metrics.
    Each chart shows trends over all available versions.
    Saves each chart as a .png in the 'visualizations' folder.
    """
    import matplotlib.pyplot as plt
    import os

    viz_folder = "visualizations"
    if os.path.exists(viz_folder):
        for f in os.listdir(viz_folder):
            os.remove(os.path.join(viz_folder, f))
    else:
        os.makedirs(viz_folder, exist_ok=True)

    # --- RELEASE SCOPE ---
    if "release_scope" in metrics:
        for metric, vdict in metrics["release_scope"].items():
            versions = sorted(vdict.keys())
            atls = [(vdict[v].get("ATLs Fixed", 0) or 0) for v in versions]
            btls = [(vdict[v].get("BTLs Fixed", 0) or 0) for v in versions]
            plt.figure()
            plt.plot(versions, atls, marker='o', label="ATLs Fixed")
            plt.plot(versions, btls, marker='o', label="BTLs Fixed")
            plt.title(f"{metric} - ATLs & BTLs Fixed Trend")
            plt.xlabel("Version")
            plt.ylabel("Fixed Count")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(viz_folder, f"{metric.replace(' ', '_').lower()}_release_scope_trend.png"))
            plt.close()

    # --- CRITICAL METRICS ---
    if "critical_metrics" in metrics:
        for metric, vdict in metrics["critical_metrics"].items():
            versions = sorted(vdict.keys())
            atls = [(vdict[v].get("ATLs Fixed", 0) or 0) for v in versions]
            btls = [(vdict[v].get("BTLs Fixed", 0) or 0) for v in versions]
            plt.figure()
            plt.plot(versions, atls, marker='o', label="ATLs")
            plt.plot(versions, btls, marker='o', label="BTLs")
            plt.title(f"{metric} - ATLs & BTLs Trend")
            plt.xlabel("Version")
            plt.ylabel("Count")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(viz_folder, f"{metric.replace(' ', '_').lower()}_critical_metrics_trend.png"))
            plt.close()

    return True


async def run_full_analysis_wst(request) -> AnalysisResponse:
    folder_path = convert_windows_path(request.folder_path)
    folder_path = os.path.normpath(folder_path)

    if not os.path.exists(folder_path):
        raise HTTPException(status_code=400, detail=f"Folder path does not exist: {folder_path}")

    pdf_files = get_pdf_files_from_folder(folder_path)
    logger.info(f"Processing {len(pdf_files)} PDF files for WST")

    # Extract versions from PDF filenames
    versions = []
    for pdf_path in pdf_files:
        match = re.search(r'(\d+\.\d+\.\d+\.\d+)', os.path.basename(pdf_path))
        if match:
            versions.append(match.group(1))
    versions = sorted(set(versions))
    if len(versions) < 2:
        raise HTTPException(status_code=400, detail="At least two versions are required for analysis")

    # Extract text for all versions
    combined_text = ""
    for pdf_path, version in zip(pdf_files, versions):
        version_text = extract_tables_text_for_version(pdf_path, version, table_indices)
        combined_text += version_text + "\n\n"
    combined_text = combined_text.strip()

    logger.info(f"Combined text sent to data_crew: {repr(combined_text)[:2000]}") 
    
    # --- Hyperlink extraction ---
    all_hyperlinks = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        hyperlink_futures = {executor.submit(extract_hyperlinks_from_pdf, pdf): pdf for pdf in pdf_files}
        for future in as_completed(hyperlink_futures):
            pdf = hyperlink_futures[future]
            try:
                all_hyperlinks.extend(future.result())
            except Exception as e:
                logger.error(f"Failed to process hyperlinks from {pdf}: {str(e)}")
                continue

    # Setup crews
    data_crew, report_crew, viz_crew, brief_summary_crew = setup_crew_wst(combined_text, versions)

    # Run data crew first
    logger.info("Starting data_crew")
    await data_crew.kickoff_async()
    logger.info("Data_crew completed")

    # --- Get and clean raw output for JSON parse ---
    raw_metrics = None
    if hasattr(data_crew.tasks[0], "output"):
        raw_metrics = getattr(data_crew.tasks[0].output, "raw", None)
        logger.info(f"Data_crew raw output type: {type(raw_metrics)}")
        logger.info(f"Data_crew raw output repr: {repr(raw_metrics)}")
    else:
        logger.warning("Data_crew.tasks[0] has no 'output' attribute.")

    if not raw_metrics or not str(raw_metrics).strip():
        logger.error("LLM did not return any output for data_crew. Check your API key, model deployment, or input prompt length.")
        raise HTTPException(status_code=500, detail="LLM did not return any output for data crew.")

    # --- CLEAN THE OUTPUT: Remove markdown code fences and language markers! ---
    metrics_str = raw_metrics.strip()
    if metrics_str.startswith("```"):
        # Remove leading ```
        metrics_str = metrics_str.lstrip("`").strip()
        # Remove "json" or "JSON" language marker if present
        if metrics_str.lower().startswith("json"):
            metrics_str = metrics_str[4:].strip()
        # Now, strip up to the first '{'
        brace_index = metrics_str.find('{')
        if brace_index >= 0:
            metrics_str = metrics_str[brace_index:]
    if metrics_str.endswith("```"):
        metrics_str = metrics_str[:-3].rstrip()

    # Try to parse as JSON and assign to shared_state.metrics
    try:
        shared_state.metrics = json.loads(metrics_str)
    except Exception as e:
        logger.error(f"Error parsing data_crew output as JSON: {e}")
        logger.error(f"Data_crew raw output that failed to parse: {repr(metrics_str)}")
        shared_state.metrics = None
        raise HTTPException(status_code=500, detail="Failed to parse LLM output as JSON. See logs for details.")

    if not shared_state.metrics or not isinstance(shared_state.metrics, dict):
        logger.error(f"Invalid metrics in shared_state: {shared_state.metrics}")
        raise HTTPException(status_code=500, detail="Failed to generate valid metrics data")

    # --- NORMALIZE THE METRICS STRUCTURE HERE ---
    shared_state.metrics = normalize_metrics(shared_state.metrics)

    # Run other crews in parallel
    logger.info("Starting report_crew, viz_crew, brief_summary_crew")
    await asyncio.gather(
        report_crew.kickoff_async(),
        viz_crew.kickoff_async(),
        brief_summary_crew.kickoff_async()
    )
    logger.info("Other crews completed")

    # Validate outputs
    if not hasattr(report_crew.tasks[-1], 'output') or not hasattr(report_crew.tasks[-1].output, 'raw'):
        logger.error("Report crew did not produce a valid output")
        raise HTTPException(status_code=500, detail="No valid report output")
    if not hasattr(viz_crew.tasks[0], 'output') or not hasattr(viz_crew.tasks[0].output, 'raw'):
        logger.error("Viz crew did not produce a valid output")
        raise HTTPException(status_code=500, detail="No valid visualization output")

    brief_summary = ""
    if hasattr(brief_summary_crew.tasks[0], 'output') and hasattr(brief_summary_crew.tasks[0].output, 'raw'):
        brief_summary = brief_summary_crew.tasks[0].output.raw.strip()
    else:
        brief_summary = "Brief summary could not be generated."

    metrics = shared_state.metrics

    # Enhance report markdown
    enhanced_report = enhance_report_markdown(report_crew.tasks[-1].output.raw)

    # Visualization
    viz_folder = "visualizations"
    if os.path.exists(viz_folder):
        shutil.rmtree(viz_folder)
    os.makedirs(viz_folder, exist_ok=True)

    script_path = "visualizations.py"
    raw_script = viz_crew.tasks[0].output.raw
    clean_script = re.sub(r'```python|```$', '', raw_script, flags=re.MULTILINE).strip()

    try:
        with shared_state.viz_lock:
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(clean_script)
            logger.info(f"Visualization script written to {script_path}")
            runpy.run_path(script_path, init_globals={'metrics': metrics})
            logger.info("Visualization script executed successfully")
    except Exception as e:
        logger.error(f"Visualization script failed: {str(e)}")
        logger.info("Running fallback visualization")
        run_fallback_visualization(metrics)

    # Collect visualization images as base64
    viz_base64 = []
    min_visualizations = 3
    if os.path.exists(viz_folder):
        viz_files = sorted([f for f in os.listdir(viz_folder) if f.endswith('.png')])
        for img in viz_files:
            img_path = os.path.join(viz_folder, img)
            base64_str = get_base64_image(img_path)
            if base64_str:
                viz_base64.append(base64_str)
        if len(viz_base64) < min_visualizations:
            logger.warning("Insufficient visualizations, running fallback")
            run_fallback_visualization(metrics)
            viz_files = sorted([f for f in os.listdir(viz_folder) if f.endswith('.png')])
            viz_base64 = []
            for img in viz_files:
                img_path = os.path.join(viz_folder, img)
                base64_str = get_base64_image(img_path)
                if base64_str:
                    viz_base64.append(base64_str)
            if len(viz_base64) < min_visualizations:
                logger.error(f"Still too few visualizations: {len(viz_base64)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to generate minimum required visualizations: got {len(viz_base64)}, need at least {min_visualizations}"
                )

    evaluation = evaluate_with_llm_judge(combined_text, enhanced_report)

    return AnalysisResponse(
        metrics=metrics,
        visualizations=viz_base64,
        report=enhanced_report,
        evaluation=evaluation,
        hyperlinks=all_hyperlinks,
        brief_summary=brief_summary
    )
