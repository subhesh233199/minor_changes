def extract_tables_text_for_version(pdf_path, version, indices_dict):
    """
    Extracts named tables as strings from a PDF, prepends version label,
    and returns a single combined string in the expected order/format.
    """
    import pdfplumber
    import pandas as pd

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
