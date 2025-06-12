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
