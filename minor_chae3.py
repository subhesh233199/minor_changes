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
