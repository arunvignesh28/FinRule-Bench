# extract json files from images

import base64
import os
import json
from openai import OpenAI

API_KEY = "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"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_INPUT_DIR = os.path.join(SCRIPT_DIR, "final_data_images")
OUTPUT_FOLDER_NAME = "raw_table_data_v2"

try:
    client = OpenAI(
      base_url="https://openrouter.ai/api/v1",
      api_key=API_KEY,
    )
except Exception as e:
    print(f"Error: Failed to initialize OpenAI client. {e}")
    exit()

def get_image_media_type(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    if ext in [".jpg", ".jpeg"]:
        return "image/jpeg"
    elif ext == ".png":
        return "image/png"
    elif ext == ".gif":
        return "image/gif"
    elif ext == ".webp":
        return "image/webp"
    else:
        return "application/octet-stream"

def encode_image_to_base64(filepath):
    with open(filepath, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def analyze_image(filepath, openai_client):
    base64_image = encode_image_to_base64(filepath)
    media_type = get_image_media_type(filepath)
    
    prompt_text = """
You are an expert OCR model for financial documents. Your task is to extract the tabular data from the provided image and format it into a specific string structure.

Follow these formatting rules precisely:
1.  Start the entire output with `[Tab]` followed by the table's title and two newlines.
2.  For each distinct time period or year column in the table, start a new section with `[Time]: <Date or Year> [SEP]` followed by a newline.
3.  Transcribe each row of the table. Format each row as `[row X]: <Cell 1 content> | <Cell 2 content> | ... [SEP]` where `X` is the row number starting from 0 for each time period.
4.  Use a pipe symbol `|` to separate columns within a row.
5.  End each row with the `[SEP]` token followed by a newline.
6.  If a cell is empty, represent it with an empty string between separators (e.g., `| |`).
7.  Combine everything into a single, continuous string. Do not wrap the output in a list or JSON.
"""
    
    try:
        completion = openai_client.chat.completions.create(
            extra_headers={},
            model="openrouter/horizon-beta",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{media_type};base64,{base64_image}"}
                        }
                    ]
                }
            ],
            timeout=120
        )
        return {"success": completion.choices[0].message.content}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    ROOT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, OUTPUT_FOLDER_NAME)

    if not os.path.exists(ROOT_INPUT_DIR) or not os.path.isdir(ROOT_INPUT_DIR):
        print(f"Error: Root input directory not found at '{ROOT_INPUT_DIR}'")
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

                image_files = [f for f in os.listdir(company_input_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp'))]
                
                total_files = len(image_files)
                if total_files == 0:
                    print(f"  -> No image files found in '{company_name}'.")
                    print("-" * 30)
                    continue

                for i, filename in enumerate(image_files):
                    full_image_path = os.path.join(company_input_path, filename)
                    print(f"  -> [{i+1}/{total_files}] Analyzing: {filename}")
                    
                    result = analyze_image(full_image_path, client)
                    
                    if "error" in result:
                        print(f"    -> FAILED: {result['error']}")
                    else:
                        try:
                            formatted_text = result['success']
                            output_filename = os.path.splitext(filename)[0] + ".json"
                            output_filepath = os.path.join(company_output_path, output_filename)
                            
                            with open(output_filepath, 'w', encoding='utf-8') as f:
                                json.dump([formatted_text], f, ensure_ascii=False, indent=4)

                            print(f"    -> SUCCESS: Saved to {os.path.join(company_name, output_filename)}")
                        except (KeyError, IndexError) as e:
                            print(f"    -> FAILED: Could not process the API response. Error: {e}")
                            print(f"    -> Full Response: {result}")
                
                print("-" * 30)



# /home/changhu/anaconda3/envs/audit/bin/python /home/changhu/auditbench-v2/extract/extract.py

# import base64
# import os
# import json
# from openai import OpenAI

# API_KEY = ""

# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# ROOT_INPUT_DIR = os.path.join(SCRIPT_DIR, "final_data_images")
# OUTPUT_FOLDER_NAME = "raw_table_data"

# try:
#     client = OpenAI(
#       base_url="https://openrouter.ai/api/v1",
#       api_key=API_KEY,
#     )
# except Exception as e:
#     print(f"Error: Failed to initialize OpenAI client. {e}")
#     exit()

# def get_image_media_type(filepath):
#     ext = os.path.splitext(filepath)[1].lower()
#     if ext in [".jpg", ".jpeg"]:
#         return "image/jpeg"
#     elif ext == ".png":
#         return "image/png"
#     elif ext == ".gif":
#         return "image/gif"
#     elif ext == ".webp":
#         return "image/webp"
#     else:
#         return "application/octet-stream"

# def encode_image_to_base64(filepath):
#     with open(filepath, "rb") as image_file:
#         return base64.b64encode(image_file.read()).decode('utf-8')

# def analyze_image(filepath, openai_client):
#     base64_image = encode_image_to_base64(filepath)
#     media_type = get_image_media_type(filepath)
    
#     prompt_text = """
# You are an expert OCR model for financial documents. Your task is to extract the tabular data from the provided image and format it into a specific string structure.

# Follow these formatting rules precisely:
# 1.  Start the entire output with `[Tab]` followed by the table's title and two newlines.
# 2.  For each distinct time period or year column in the table, start a new section with `[Time]: <Date or Year> [SEP]` followed by a newline.
# 3.  Transcribe each row of the table. Format each row as `[row X]: <Cell 1 content> | <Cell 2 content> | ... [SEP]` where `X` is the row number starting from 0 for each time period.
# 4.  Use a pipe symbol `|` to separate columns within a row.
# 5.  End each row with the `[SEP]` token followed by a newline.
# 6.  If a cell is empty, represent it with an empty string between separators (e.g., `| |`).
# 7.  Combine everything into a single, continuous string. Do not wrap the output in a list or JSON.
# """
    
#     try:
#         completion = openai_client.chat.completions.create(
#             extra_headers={},
#             model="qwen/qwen2.5-vl-72b-instruct:free",
#             messages=[
#                 {
#                     "role": "user",
#                     "content": [
#                         {"type": "text", "text": prompt_text},
#                         {
#                             "type": "image_url",
#                             "image_url": {"url": f"data:{media_type};base64,{base64_image}"}
#                         }
#                     ]
#                 }
#             ],
#             timeout=120
#         )
#         return {"success": completion.choices[0].message.content}
#     except Exception as e:
#         return {"error": str(e)}

# if __name__ == "__main__":
#     ROOT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, OUTPUT_FOLDER_NAME)

#     if not os.path.exists(ROOT_INPUT_DIR) or not os.path.isdir(ROOT_INPUT_DIR):
#         print(f"Error: Root input directory not found at '{ROOT_INPUT_DIR}'")
#     else:
#         print(f"Input Directory:  {ROOT_INPUT_DIR}")
#         print(f"Output Directory: {ROOT_OUTPUT_DIR}\n")
#         os.makedirs(ROOT_OUTPUT_DIR, exist_ok=True)

#         all_companies = sorted(os.listdir(ROOT_INPUT_DIR))
#         start_after_company = "Bilibili"
        
#         try:
#             start_index = all_companies.index(start_after_company) + 1
#         except ValueError:
#             print(f"Warning: Company '{start_after_company}' not found. Starting from the beginning.")
#             start_index = 0

#         companies_to_process = all_companies[start_index:]

#         if not companies_to_process:
#             print(f"No new companies to process after '{start_after_company}'.")

#         for company_name in companies_to_process:
#             company_input_path = os.path.join(ROOT_INPUT_DIR, company_name)
            
#             if os.path.isdir(company_input_path):
#                 print(f"Processing Company: {company_name}")
                
#                 output_company_name = company_name.replace('&', 'and')
#                 if output_company_name != company_name:
#                     print(f"  -> Sanitized output folder name to: '{output_company_name}'")

#                 company_output_path = os.path.join(ROOT_OUTPUT_DIR, output_company_name)
#                 os.makedirs(company_output_path, exist_ok=True)

#                 image_files = [f for f in os.listdir(company_input_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp'))]
                
#                 total_files = len(image_files)
#                 if total_files == 0:
#                     print(f"  -> No image files found in '{company_name}'.")
#                     print("-" * 30)
#                     continue

#                 for i, filename in enumerate(image_files):
#                     full_image_path = os.path.join(company_input_path, filename)
#                     print(f"  -> [{i+1}/{total_files}] Analyzing: {filename}")
                    
#                     result = analyze_image(full_image_path, client)
                    
#                     if "error" in result:
#                         print(f"    -> FAILED: {result['error']}")
#                     else:
#                         try:
#                             formatted_text = result['success']
#                             output_filename = os.path.splitext(filename)[0] + ".json"
#                             output_filepath = os.path.join(company_output_path, output_filename)
                            
#                             with open(output_filepath, 'w', encoding='utf-8') as f:
#                                 json.dump([formatted_text], f, ensure_ascii=False, indent=4)

#                             print(f"    -> SUCCESS: Saved to {os.path.join(output_company_name, output_filename)}")
#                         except (KeyError, IndexError) as e:
#                             print(f"    -> FAILED: Could not process the API response. Error: {e}")
#                             print(f"    -> Full Response: {result}")
                
#                 print("-" * 30)