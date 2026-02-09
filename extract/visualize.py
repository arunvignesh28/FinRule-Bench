# markdown visualization for tables

import os
import json
from openai import OpenAI

API_KEY = "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"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_INPUT_DIR = os.path.join(SCRIPT_DIR, "raw_table_data_v2")
OUTPUT_FOLDER_NAME = "visual_tables_md_v2"
ROOT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, OUTPUT_FOLDER_NAME)

try:
    client = OpenAI(
      base_url="https://openrouter.ai/api/v1",
      api_key=API_KEY,
    )
except Exception as e:
    print(f"Error: Failed to initialize OpenAI client. {e}")
    exit()

def generate_markdown_table(text_content, openai_client):
    prompt_text = f"""
You are an expert data converter. Your task is to convert the provided structured text into a clean and readable Markdown file.

The input text uses this format:
- `[Tab]` contains the document's main title.
- `[Time]: <Date or Year> [SEP]` is the subtitle for the table.
- `[row X]: <Cell 1> | <Cell 2> | ... [SEP]` represents a table row.

Your output MUST be only in Markdown format.
1.  Use the text from `[Tab]` as a Level 1 Markdown Header (e.g., `# Main Title`).
2.  Use the text from `[Time]` as a Level 3 Markdown Header (e.g., `### Table Subtitle`).
3.  Convert the `[row]` data into a standard Markdown table. The first row (`[row 0]`) is the table header.
4.  Ensure the Markdown table is correctly formatted with `|` separators and a header separation line (`|---|---|...`).

Your response must be ONLY the final Markdown text. Do not include any explanations, comments, or any text other than the Markdown code itself.

Here is the data to convert:
---
{text_content}
---
"""
    
    try:
        completion = openai_client.chat.completions.create(
            model="moonshotai/kimi-k2:free",
            messages=[
                {
                    "role": "user",
                    "content": prompt_text
                }
            ],
            timeout=120
        )
        return {"success": completion.choices[0].message.content}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    if not os.path.exists(ROOT_INPUT_DIR) or not os.path.isdir(ROOT_INPUT_DIR):
        print(f"Error: Root input directory not found at '{ROOT_INPUT_DIR}'")
        print("Please make sure the 'raw_table_data' folder exists.")
    else:
        print(f"Input Directory:  {ROOT_INPUT_DIR}")
        print(f"Output Directory: {ROOT_OUTPUT_DIR}\n")
        os.makedirs(ROOT_OUTPUT_DIR, exist_ok=True)

        for company_name in os.listdir(ROOT_INPUT_DIR):
            company_input_path = os.path.join(ROOT_INPUT_DIR, company_name)
            
            if os.path.isdir(company_input_path):
                print(f"Processing Company: {company_name}")
                
                company_output_path = os.path.join(ROOT_OUTPUT_DIR, company_name)
                os.makedirs(company_output_path, exist_ok=True)

                json_files = [f for f in os.listdir(company_input_path) if f.lower().endswith('.json')]
                
                total_files = len(json_files)
                if total_files == 0:
                    print(f"  -> No JSON files found in '{company_name}'.")
                    print("-" * 30)
                    continue

                for i, filename in enumerate(json_files):
                    full_json_path = os.path.join(company_input_path, filename)
                    print(f"  -> [{i+1}/{total_files}] Generating Markdown for: {filename}")
                    
                    try:
                        with open(full_json_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            original_text = data[0]

                        result = generate_markdown_table(original_text, client)
                        
                        if "error" in result:
                            print(f"    -> FAILED (API Error): {result['error']}")
                        else:
                            markdown_content = result['success']
                            output_filename = os.path.splitext(filename)[0] + ".md"
                            output_filepath = os.path.join(company_output_path, output_filename)
                            
                            with open(output_filepath, 'w', encoding='utf-8') as f:
                                f.write(markdown_content)

                            print(f"    -> SUCCESS: Saved to {os.path.join(company_name, output_filename)}")

                    except Exception as e:
                        print(f"    -> FAILED (File/JSON Error): {e}")
                
                print("-" * 30)





# /home/changhu/anaconda3/envs/audit/bin/python /home/changhu/auditbench-v2/extract/visualize.py

# import os
# import json
# from openai import OpenAI

# API_KEY = ""

# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  
# ROOT_INPUT_DIR = os.path.join(SCRIPT_DIR, "raw_table_data")
# OUTPUT_FOLDER_NAME = "visual_tables_md"
# ROOT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, OUTPUT_FOLDER_NAME)

# try:
#     client = OpenAI(
#       base_url="https://openrouter.ai/api/v1",
#       api_key=API_KEY,
#     )
# except Exception as e:
#     print(f"Error: Failed to initialize OpenAI client. {e}")
#     exit()

# def generate_markdown_table(text_content, openai_client):
#     prompt_text = f"""
# You are an expert data converter. Your task is to convert the provided structured text into a clean and readable Markdown file.

# The input text uses this format:
# - `[Tab]` contains the document's main title.
# - `[Time]: <Date or Year> [SEP]` is the subtitle for the table.
# - `[row X]: <Cell 1> | <Cell 2> | ... [SEP]` represents a table row.

# Your output MUST be only in Markdown format.
# 1.  Use the text from `[Tab]` as a Level 1 Markdown Header (e.g., `# Main Title`).
# 2.  Use the text from `[Time]` as a Level 3 Markdown Header (e.g., `### Table Subtitle`).
# 3.  Convert the `[row]` data into a standard Markdown table. The first row (`[row 0]`) is the table header.
# 4.  Ensure the Markdown table is correctly formatted with `|` separators and a header separation line (`|---|---|...`).

# Your response must be ONLY the final Markdown text. Do not include any explanations, comments, or any text other than the Markdown code itself.

# Here is the data to convert:
# ---
# {text_content}
# ---
# """
    
#     try:
#         completion = openai_client.chat.completions.create(
#             model="moonshotai/kimi-k2:free",
#             messages=[
#                 {
#                     "role": "user",
#                     "content": prompt_text
#                 }
#             ],
#             timeout=120
#         )
#         return {"success": completion.choices[0].message.content}
#     except Exception as e:
#         return {"error": str(e)}

# if __name__ == "__main__":
#     if not os.path.exists(ROOT_INPUT_DIR) or not os.path.isdir(ROOT_INPUT_DIR):
#         print(f"Error: Root input directory not found at '{ROOT_INPUT_DIR}'")
#         print("Please make sure the 'raw_table_data' folder exists.")
#     else:
#         print(f"Input Directory:  {ROOT_INPUT_DIR}")
#         print(f"Output Directory: {ROOT_OUTPUT_DIR}\n")
#         os.makedirs(ROOT_OUTPUT_DIR, exist_ok=True)

#         all_companies = sorted(os.listdir(ROOT_INPUT_DIR))
        
#         start_after_company = "Bilibili" 
        
#         try:
#             start_index = all_companies.index(start_after_company) + 1
#             print(f"Starting processing after company: '{start_after_company}'")
#         except ValueError:
#             print(f"Warning: Company '{start_after_company}' not found. Starting from the beginning.")
#             start_index = 0

#         companies_to_process = all_companies[start_index:]

#         if not companies_to_process:
#             print(f"No new companies to process after '{start_after_company}'.")
#         else:
#             print(f"Found {len(companies_to_process)} companies to process.\n")

#         for company_name in companies_to_process:
#             company_input_path = os.path.join(ROOT_INPUT_DIR, company_name)
            
#             if os.path.isdir(company_input_path):
#                 print(f"Processing Company: {company_name}")
                
#                 output_company_name = company_name.replace('&', 'and')
#                 if output_company_name != company_name:
#                     print(f"  -> Sanitized output folder name to: '{output_company_name}'")
                
#                 company_output_path = os.path.join(ROOT_OUTPUT_DIR, output_company_name)
#                 os.makedirs(company_output_path, exist_ok=True)

#                 json_files = [f for f in os.listdir(company_input_path) if f.lower().endswith('.json')]
                
#                 total_files = len(json_files)
#                 if total_files == 0:
#                     print(f"  -> No JSON files found in '{company_name}'.")
#                     print("-" * 30)
#                     continue

#                 for i, filename in enumerate(json_files):
#                     full_json_path = os.path.join(company_input_path, filename)
#                     print(f"  -> [{i+1}/{total_files}] Generating Markdown for: {filename}")
                    
#                     try:
#                         with open(full_json_path, 'r', encoding='utf-8') as f:
#                             data = json.load(f)
#                             original_text = data[0]

#                         result = generate_markdown_table(original_text, client)
                        
#                         if "error" in result:
#                             print(f"    -> FAILED (API Error): {result['error']}")
#                         else:
#                             markdown_content = result['success']
#                             output_filename = os.path.splitext(filename)[0] + ".md"
#                             output_filepath = os.path.join(company_output_path, output_filename)
                            
#                             with open(output_filepath, 'w', encoding='utf-8') as f:
#                                 f.write(markdown_content)

#                             print(f"    -> SUCCESS: Saved to {os.path.join(output_company_name, output_filename)}")

#                     except Exception as e:
#                         print(f"    -> FAILED (File/JSON Error): {e}")
                
#                 print("-" * 30)