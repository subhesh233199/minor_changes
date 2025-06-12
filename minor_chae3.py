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
