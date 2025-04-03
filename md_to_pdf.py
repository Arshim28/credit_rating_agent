import argparse
import markdown
import os
from weasyprint import HTML, CSS
from weasyprint.text.fonts import FontConfiguration

def convert_markdown_to_pdf(input_file, output_file=None, title="Placement Memorandum Summary"):
    """
    Convert a markdown file to PDF
    
    Args:
        input_file: Path to the markdown file
        output_file: Path to save the PDF file (default: same name with .pdf extension)
        title: Title to use in the PDF
    """
    # Default output filename if not specified
    if output_file is None:
        output_file = os.path.splitext(input_file)[0] + '.pdf'
    
    # Read the markdown content
    with open(input_file, 'r', encoding='utf-8') as f:
        markdown_content = f.read()
    
    # Convert markdown to HTML
    html_content = markdown.markdown(markdown_content, extensions=['tables', 'fenced_code'])
    
    # Add some CSS styling
    css_content = """
    @page {
        margin: 1cm;
        @top-center {
            content: "RiskX";
            font-size: 10pt;
            color: #666;
        }
        @bottom-center {
            content: "Page " counter(page) " of " counter(pages);
            font-size: 10pt;
            color: #666;
        }
    }
    body {
        font-family: Arial, sans-serif;
        font-size: 12pt;
        line-height: 1.4;
        color: #333;
    }
    h1 {
        font-size: 20pt;
        color: #003366;
        page-break-before: always;
        margin-top: 1cm;
    }
    h1:first-of-type {
        page-break-before: avoid;
    }
    h2 {
        font-size: 16pt;
        color: #003366;
        margin-top: 0.7cm;
    }
    h3 {
        font-size: 14pt;
        color: #003366;
        margin-top: 0.5cm;
    }
    ul, ol {
        margin-left: 0.7cm;
    }
    table {
        border-collapse: collapse;
        width: 100%;
        margin: 1em 0;
    }
    th, td {
        border: 1px solid #ddd;
        padding: 8px;
    }
    th {
        background-color: #f2f2f2;
        font-weight: bold;
    }
    code {
        font-family: monospace;
        background-color: #f5f5f5;
        padding: 2px 4px;
        border-radius: 4px;
    }
    pre {
        background-color: #f5f5f5;
        padding: 10px;
        border-radius: 4px;
        overflow-x: auto;
    }
    .page-break {
        page-break-after: always;
    }
    """
    
    # Complete HTML document
    complete_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>{title}</title>
        <style>
            {css_content}
        </style>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """
    
    # Configure fonts
    font_config = FontConfiguration()
    
    # Create PDF
    print(f"Converting {input_file} to PDF...")
    HTML(string=complete_html).write_pdf(
        output_file,
        stylesheets=[CSS(string=css_content)],
        font_config=font_config
    )
    
    print(f"PDF saved to {output_file}")
    return output_file

def main():
    parser = argparse.ArgumentParser(description='Convert markdown file to PDF')
    parser.add_argument('input_file', help='Path to markdown file')
    parser.add_argument('--output', '-o', help='Output PDF file path')
    parser.add_argument('--title', '-t', default='Placement Memorandum Summary', 
                        help='Title for the PDF document')
    
    args = parser.parse_args()
    
    convert_markdown_to_pdf(args.input_file, args.output, args.title)

if __name__ == '__main__':
    main()