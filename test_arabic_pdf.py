#!/usr/bin/env python3
# Test Arabic PDF generation

import arabic_reshaper
from bidi.algorithm import get_display
import weasyprint
import html

def test_arabic_processing():
    # Test Arabic text
    arabic_text = 'مرحبا بكم في اجتماعنا اليوم'
    print('Original:', arabic_text)
    
    # Process Arabic text
    reshaped = arabic_reshaper.reshape(arabic_text)
    bidi_text = get_display(reshaped)
    print('Processed:', bidi_text)
    
    # Test HTML generation
    html_content = f'''
    <!DOCTYPE html>
    <html lang="ar" dir="rtl">
    <head>
        <meta charset="UTF-8">
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Noto+Naskh+Arabic:wght@400;700&display=swap');
            body {{
                font-family: 'Noto Naskh Arabic', Arial, sans-serif;
                direction: rtl;
                text-align: right;
                line-height: 1.6;
                margin: 20px;
            }}
            h1 {{
                font-size: 18px;
                color: #333;
            }}
        </style>
    </head>
    <body>
        <h1>{html.escape(bidi_text)}</h1>
        <p>Testing Arabic in PDF</p>
        <p>Speaker 1: {html.escape(bidi_text)}</p>
        <p>[00:00:05–00:00:14] Speaker 1: هذا اختبار للنص العربي</p>
    </body>
    </html>
    '''
    
    try:
        # Generate test PDF
        weasyprint.HTML(string=html_content).write_pdf('/tmp/test_arabic.pdf')
        print('✅ PDF generated successfully at /tmp/test_arabic.pdf')
    except Exception as e:
        print('❌ Error generating PDF:', e)

if __name__ == "__main__":
    test_arabic_processing()
