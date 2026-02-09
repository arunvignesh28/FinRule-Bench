#!/usr/bin/env python3
"""
Data Comparison Tool - For comparing image files and Markdown files
This tool compares corresponding files in final_data_images and visual_tables_md_v2 folders
"""

import os
import re
import base64
from pathlib import Path
from typing import Dict, List, Tuple, Set
import markdown
import webbrowser
from datetime import datetime

class DataComparisonTool:
    def __init__(self, base_path: str):
        """
        Initialize comparison tool
        
        Args:
            base_path: Path to the extract folder
        """
        self.base_path = Path(base_path)
        self.images_path = self.base_path / "final_data_images"
        self.markdown_path = self.base_path / "visual_tables_md_v2"
        self.output_dir = self.base_path / "comparison_output"
        
        # Ensure output directory exists
        self.output_dir.mkdir(exist_ok=True)
        
    def get_company_folders(self) -> Set[str]:
        """Get all company folder names"""
        image_companies = set()
        md_companies = set()
        
        if self.images_path.exists():
            image_companies = {f.name for f in self.images_path.iterdir() if f.is_dir()}
        
        if self.markdown_path.exists():
            md_companies = {f.name for f in self.markdown_path.iterdir() if f.is_dir()}
            
        # Return intersection to ensure both sides have corresponding folders
        return image_companies.intersection(md_companies)
    
    def get_file_pairs(self, company: str) -> List[Tuple[str, str, str]]:
        """
        Get file pairs for a specific company folder
        
        Returns:
            List of tuples: (base_name, image_path, markdown_path)
        """
        image_dir = self.images_path / company
        md_dir = self.markdown_path / company
        
        file_pairs = []
        
        if not (image_dir.exists() and md_dir.exists()):
            return file_pairs
            
        # Get image files
        image_files = {}
        for img_file in image_dir.iterdir():
            if img_file.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                base_name = img_file.stem
                image_files[base_name] = img_file
        
        # Get markdown files and match them
        for md_file in md_dir.iterdir():
            if md_file.suffix.lower() == '.md':
                base_name = md_file.stem
                if base_name in image_files:
                    file_pairs.append((base_name, str(image_files[base_name]), str(md_file)))
        
        return sorted(file_pairs)
    
    def image_to_base64(self, image_path: str) -> str:
        """Convert image file to base64 encoding"""
        try:
            with open(image_path, 'rb') as f:
                img_data = f.read()
                img_base64 = base64.b64encode(img_data).decode('utf-8')
                
                # Determine MIME type
                ext = Path(image_path).suffix.lower()
                if ext == '.png':
                    mime_type = 'image/png'
                elif ext in ['.jpg', '.jpeg']:
                    mime_type = 'image/jpeg'
                else:
                    mime_type = 'image/png'  # default
                    
                return f"data:{mime_type};base64,{img_base64}"
        except Exception as e:
            print(f"Error encoding image {image_path}: {e}")
            return ""
    
    def read_markdown_content(self, md_path: str) -> str:
        """Read Markdown file content"""
        try:
            with open(md_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"Error reading markdown {md_path}: {e}")
            return f"Error reading file: {e}"
    
    def generate_html_report(self, company: str = None) -> str:
        """
        Generate HTML comparison report
        
        Args:
            company: Specify company name, if None generate report for all companies
            
        Returns:
            Path of generated HTML file
        """
        companies = [company] if company else sorted(self.get_company_folders())
        
        if not companies:
            print("No matching companies found!")
            return ""
        
        # Generate HTML content
        html_content = self._generate_html_header()
        
        for comp in companies:
            file_pairs = self.get_file_pairs(comp)
            if file_pairs:
                html_content += self._generate_company_section(comp, file_pairs)
        
        html_content += self._generate_html_footer()
        
        # Save HTML file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if company:
            filename = f"comparison_{company.replace(' ', '_')}_{timestamp}.html"
        else:
            filename = f"comparison_all_{timestamp}.html"
            
        output_file = self.output_dir / filename
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        print(f"HTML report generated: {output_file}")
        return str(output_file)
    
    def _generate_html_header(self) -> str:
        """Generate HTML header"""
        return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Data Comparison Report</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
            line-height: 1.6;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        
        .header h1 {
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
        }
        
        .header p {
            margin: 10px 0 0 0;
            opacity: 0.9;
            font-size: 1.1em;
        }
        
        .company-section {
            margin: 40px 0;
            border-bottom: 3px solid #eee;
            padding-bottom: 40px;
        }
        
        .company-title {
            background: #f8f9fa;
            padding: 20px 30px;
            margin: 0 0 30px 0;
            border-left: 5px solid #667eea;
        }
        
        .company-title h2 {
            margin: 0;
            color: #333;
            font-size: 1.8em;
        }
        
        .file-comparison {
            margin: 30px 0;
            border: 1px solid #ddd;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }
        
        .file-header {
            background: #667eea;
            color: white;
            padding: 15px 20px;
            font-size: 1.2em;
            font-weight: 500;
        }
        
        .comparison-container {
            display: flex;
            min-height: 600px;
        }
        
        .image-panel, .markdown-panel {
            flex: 1;
            padding: 20px;
            overflow: auto;
        }
        
        .image-panel {
            background: #fafafa;
            border-right: 2px solid #eee;
            text-align: center;
        }
        
        .markdown-panel {
            background: white;
        }
        
        .panel-title {
            font-size: 1.1em;
            font-weight: 600;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid #eee;
            color: #555;
        }
        
        .comparison-image {
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 5px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        
        .markdown-content {
            font-size: 0.95em;
            line-height: 1.6;
        }
        
        .markdown-content table {
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
            font-size: 0.9em;
        }
        
        .markdown-content th,
        .markdown-content td {
            border: 1px solid #ddd;
            padding: 8px 12px;
            text-align: left;
        }
        
        .markdown-content th {
            background-color: #f8f9fa;
            font-weight: 600;
            color: #555;
        }
        
        .markdown-content tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        
        .markdown-content h1,
        .markdown-content h2,
        .markdown-content h3 {
            color: #333;
            margin-top: 25px;
            margin-bottom: 15px;
        }
        
        .markdown-content h1 {
            font-size: 1.3em;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }
        
        .markdown-content h2 {
            font-size: 1.2em;
            color: #667eea;
        }
        
        .markdown-content h3 {
            font-size: 1.1em;
            color: #764ba2;
        }
        
        .summary {
            background: #f8f9fa;
            padding: 30px;
            margin-top: 40px;
            text-align: center;
            color: #666;
        }
        
        @media (max-width: 768px) {
            .comparison-container {
                flex-direction: column;
            }
            
            .image-panel {
                border-right: none;
                border-bottom: 2px solid #eee;
            }
            
            body {
                padding: 10px;
            }
            
            .header {
                padding: 20px;
            }
            
            .header h1 {
                font-size: 2em;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Data Comparison Report</h1>
            <p>Images vs Markdown Tables Comparison</p>
            <p>Generated on """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
        </div>
        <div class="content">
"""
    
    def _generate_company_section(self, company: str, file_pairs: List[Tuple[str, str, str]]) -> str:
        """Generate HTML section for a single company"""
        html = f"""
        <div class="company-section">
            <div class="company-title">
                <h2>🏢 {company}</h2>
            </div>
"""
        
        for base_name, image_path, md_path in file_pairs:
            # Generate base64 image
            img_base64 = self.image_to_base64(image_path)
            
            # Read markdown content and convert to HTML
            md_content = self.read_markdown_content(md_path)
            md_html = markdown.markdown(md_content, extensions=['tables'])
            
            html += f"""
            <div class="file-comparison">
                <div class="file-header">
                    📄 {base_name}
                </div>
                <div class="comparison-container">
                    <div class="image-panel">
                        <div class="panel-title">🖼️ Image View</div>
                        {"<img src='" + img_base64 + "' alt='" + base_name + "' class='comparison-image'>" if img_base64 else "<p>Failed to load image</p>"}
                    </div>
                    <div class="markdown-panel">
                        <div class="panel-title">📝 Markdown Content</div>
                        <div class="markdown-content">
                            {md_html}
                        </div>
                    </div>
                </div>
            </div>
"""
        
        html += "</div>"
        return html
    
    def _generate_html_footer(self) -> str:
        """Generate HTML footer"""
        companies_count = len(self.get_company_folders())
        total_files = sum(len(self.get_file_pairs(comp)) for comp in self.get_company_folders())
        
        return f"""
        </div>
        <div class="summary">
            <p><strong>Summary:</strong> Compared {total_files} file pairs from {companies_count} companies</p>
            <p>Generated by Data Comparison Tool • {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        </div>
    </div>
</body>
</html>
"""
    
    def list_companies(self) -> None:
        """List all available companies"""
        companies = sorted(self.get_company_folders())
        print(f"\nFound {len(companies)} companies with matching data:")
        for i, company in enumerate(companies, 1):
            file_pairs = self.get_file_pairs(company)
            print(f"{i:3d}. {company} ({len(file_pairs)} file pairs)")
    
    def compare_company(self, company: str) -> str:
        """Compare data for a single company"""
        if company not in self.get_company_folders():
            print(f"Company '{company}' not found!")
            return ""
        
        return self.generate_html_report(company)
    
    def compare_all(self) -> str:
        """Compare data for all companies"""
        return self.generate_html_report()


def main():
    """Main function"""
    # Assume script runs in extract folder
    current_dir = Path(__file__).parent
    tool = DataComparisonTool(str(current_dir))
    
    print("📊 Data Comparison Tool")
    print("=" * 50)
    
    while True:
        print("\nOptions:")
        print("1. List all companies")
        print("2. Compare specific company")
        print("3. Compare all companies")
        print("4. Exit")
        
        choice = input("\nSelect option (1-4): ").strip()
        
        if choice == '1':
            tool.list_companies()
            
        elif choice == '2':
            tool.list_companies()
            company = input("\nEnter company name (exact match): ").strip()
            if company:
                html_file = tool.compare_company(company)
                if html_file:
                    open_browser = input("Open in browser? (y/n): ").strip().lower()
                    if open_browser == 'y':
                        webbrowser.open(f'file://{html_file}')
                        
        elif choice == '3':
            print("Generating comparison for all companies...")
            html_file = tool.compare_all()
            if html_file:
                open_browser = input("Open in browser? (y/n): ").strip().lower()
                if open_browser == 'y':
                    webbrowser.open(f'file://{html_file}')
                    
        elif choice == '4':
            print("Goodbye!")
            break
            
        else:
            print("Invalid option. Please try again.")


if __name__ == "__main__":
    main()