"""Find section line numbers in persona_i18n.py"""
with open(r"src/persona_i18n.py", encoding="utf-8-sig") as f:
    lines = f.readlines()

with open("sections_output.txt", "w") as out:
    targets = [
        "PERSONA_DESCRIPTORS_I18N",
        "PERSONA_TEMPLATES_I18N",
        "PERSONA_SCAFFOLD_I18N",
        "UTILITARIAN_PERSONA_I18N",
        "COUNTRY_NATIVE_NAME",
    ]
    for i, line in enumerate(lines, 1):
        for t in targets:
            if t in line and ("Dict" in line):
                out.write(f"{t} -> line {i}\n")
    
    # Find where "fa": sections start to know the last language block
    out.write("\n--- 'fa' blocks ---\n")
    for i, line in enumerate(lines, 1):
        if line.strip().startswith('"fa"') and ('{' in line or ':' in line):
            out.write(f"line {i}: {line.strip()[:60]}\n")
    
    out.write(f"\nTotal lines: {len(lines)}\n")
print("Done - see sections_output.txt")
