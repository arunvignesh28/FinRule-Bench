import json
from pathlib import Path

ROOT_DIR = Path('multiple_combined_table_v2')
OUTPUT_FILE = Path('data.json')

merged_data = []

for company_dir in ROOT_DIR.iterdir():
    if not company_dir.is_dir():
        continue
    company_name = company_dir.name
    for json_file in company_dir.glob('*.json'):
        table_name = json_file.stem
        with json_file.open('r', encoding='utf-8') as f:
            content = json.load(f)
        record = content if isinstance(content, dict) else {'data': content}
        record['company_name'] = company_name
        record['table_name'] = table_name
        merged_data.append(record)

with OUTPUT_FILE.open('w', encoding='utf-8') as f:
    json.dump(merged_data, f, ensure_ascii=False, indent=2)

print(f'✅ Merge complete: {len(merged_data)} records saved to {OUTPUT_FILE.resolve()}')