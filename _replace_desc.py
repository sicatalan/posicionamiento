import re
p = 'app.py'
with open(p,'r',encoding='utf-8',errors='replace') as f:
    s = f.read()
# Replace subheader line
s = re.sub(r"^\s*st\.subheader\(\"Heatmap:.*\)\s*$",
           '    st.subheader("Comparacion por Familia: precios interno vs canales")',
           s, flags=re.M)
# Replace caption line that mentions ratio_vs_interno
s = re.sub(r"^\s*st\.caption\(\"ratio_vs_interno[\s\S]*?\)\s*$",
           '    st.caption("Compara precios absolutos por familia entre el canal interno y otros canales. Filtra por categoria, familia y canales.")',
           s, flags=re.M)
# Replace info block about Heatmap formulas
s = re.sub(r"^\s*st\.info\(\s*\n\s*\"Heatmap \(ratio vs interno\):\\n\"[\s\S]*?\)\s*$",
           '    st.info(\n        "Lectura:\\n"\n        "- Cada palito y marcador representa el precio agregado por canal (promedio por defecto, o mediana).\\n"\n        "- El canal interno aparece destacado y puede incluirse/excluirse en los filtros."\n    )',
           s, flags=re.M)
with open(p,'w',encoding='utf-8',newline='') as f:
    f.write(s)
print('DONE')
