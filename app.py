from flask import Flask, request, jsonify, render_template, send_file
from werkzeug.utils import secure_filename
import pdfplumber
import os
from openai import OpenAI
import httpx
from deep_translator import GoogleTranslator
from langdetect import detect
import json
import csv
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from io import BytesIO
import requests
import tempfile
import logging

# تنظیم لاگ‌گذاری
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # حداکثر 16MB

# URL فونت Vazir
VAZIR_FONT_URL = 'https://raw.githubusercontent.com/rastikerdar/vazir-font/master/dist/Vazir.ttf'

# دانلود و رجیستر فونت برای PDF
def register_vazir_font():
    try:
        response = requests.get(VAZIR_FONT_URL, timeout=10)
        response.raise_for_status()
        with tempfile.NamedTemporaryFile(delete=False, suffix='.ttf') as tmp_file:
            tmp_file.write(response.content)
            tmp_file_path = tmp_file.name
        pdfmetrics.registerFont(TTFont('Vazir', tmp_file_path))
        return tmp_file_path
    except Exception as e:
        logger.error(f"خطا در دانلود فونت Vazir: {str(e)}")
        raise Exception(f"خطا در دانلود فونت Vazir: {str(e)}")

# بررسی نوع فایل
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ایجاد پوشه آپلود
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# پرامپت استاندارد
DEFAULT_PROMPT = """
Extract the following information from the resume text and provide a structured JSON output:
- experience: Number of years of professional experience (integer, estimate if not explicit).
- skills: List of key skills (array of strings, max 10 skills).
- education: Highest degree and field of study (string, e.g., "Master's in Computer Science").
- summary: A textual summary of strengths and weaknesses in Persian (150-200 words).
Return the output in JSON format.
"""

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_resume():
    try:
        if 'resumes' not in request.files or not request.form.get('api_key'):
            logger.error("فایل رزومه یا توکن API وارد نشده است")
            return jsonify({'error': 'فایل رزومه یا توکن API وارد نشده است'}), 400
        
        files = request.files.getlist('resumes')
        api_key = request.form.get('api_key')
        custom_prompt = request.form.get('custom_prompt', DEFAULT_PROMPT)
        criteria = json.loads(request.form.get('criteria', '{}'))
        
        # اعتبارسنجی معیارها
        if not criteria:
            logger.error("هیچ معیاری تعریف نشده است")
            return jsonify({'error': 'حداقل یک معیار باید تعریف شود'}), 400
        
        # ایجاد کلاینت OpenAI
        try:
            gapgpt_client = OpenAI(
                api_key=api_key,
                base_url='https://api.gapgpt.app/v1',
                http_client=httpx.Client(follow_redirects=True)
            )
        except Exception as e:
            logger.error(f"خطا در مقداردهی کلاینت API: {str(e)}")
            return jsonify({'error': f'خطا در مقداردهی کلاینت API: {str(e)}'}), 400
        
        results = []
        
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                
                try:
                    # استخراج متن از PDF
                    with pdfplumber.open(file_path) as pdf:
                        text = ''.join(page.extract_text() or '' for page in pdf.pages)
                    
                    # تشخیص زبان
                    lang = detect(text)
                    if lang != 'fa':
                        text = GoogleTranslator(source='auto', target='fa').translate(text)
                    
                    # ارسال به API گپ‌جی‌پی‌تی
                    response = gapgpt_client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "user", "content": custom_prompt + "\n\nResume text:\n" + text}
                        ]
                    )
                    
                    analysis = json.loads(response.choices[0].message.content)
                    
                    # محاسبه امتیاز پویا
                    score = 0
                    for criterion, weight in criteria.items():
                        if criterion == 'experience':
                            score += analysis.get('experience', 0) * 10 * weight
                        elif criterion == 'skills':
                            score += len(analysis.get('skills', [])) * 5 * weight
                        elif criterion == 'education':
                            score += {'PhD': 30, 'Master': 25, 'Bachelor': 20}.get(
                                analysis.get('education', '').split()[0], 10) * weight
                        # معیارهای سفارشی در آینده قابل گسترش
                    score = min(round(score, 2), 100)
                    
                    results.append({
                        'filename': filename,
                        'analysis': analysis,
                        'score': score,
                        'summary': analysis.get('summary', 'تحلیل در دسترس نیست')
                    })
                    
                    # حذف فایل
                    os.remove(file_path)
                    
                except Exception as e:
                    logger.error(f"خطا در پردازش فایل {filename}: {str(e)}")
                    results.append({
                        'filename': filename,
                        'error': f'خطا در پردازش: {str(e)}'
                    })
            else:
                logger.error(f"فرمت فایل {filename} مجاز نیست")
                results.append({
                    'filename': filename,
                    'error': 'فرمت فایل مجاز نیست'
                })
        
        return jsonify({'results': results})
    
    except Exception as e:
        logger.error(f"خطای عمومی در /upload: {str(e)}")
        return jsonify({'error': f'خطای سرور: {str(e)}'}), 500

@app.route('/download/csv/<filename>', methods=['GET'])
def download_csv(filename):
    try:
        results = json.loads(request.args.get('results'))
        
        output = BytesIO()
        writer = csv.writer(output)
        writer.writerow(['Filename', 'Score', 'Experience', 'Skills', 'Education', 'Summary'])
        
        for result in results:
            if result['filename'] == filename and 'analysis' in result:
                writer.writerow([
                    result['filename'],
                    result['score'],
                    result['analysis'].get('experience', 'نامشخص'),
                    ', '.join(result['analysis'].get('skills', [])),
                    result['analysis'].get('education', 'نامشخص'),
                    result['summary']
                ])
        
        output.seek(0)
        return send_file(
            output,
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'{filename}_analysis.csv'
        )
    
    except Exception as e:
        logger.error(f"خطا در دانلود CSV: {str(e)}")
        return jsonify({'error': f'خطا در دانلود CSV: {str(e)}'}), 500

@app.route('/download/pdf/<filename>', methods=['GET'])
def download_pdf(filename):
    try:
        results = json.loads(request.args.get('results'))
        output = BytesIO()
        c = canvas.Canvas(output, pagesize=A4)
        
        # رجیستر فونت
        font_path = register_vazir_font()
        c.setFont('Vazir', 12)
        y = 800
        
        for result in results:
            if result['filename'] == filename and 'analysis' in result:
                c.drawString(50, y, f"رزومه: {result['filename']}")
                y -= 20
                c.drawString(50, y, f"امتیاز: {result['score']}")
                y -= 20
                c.drawString(50, y, f"تجربه: {result['analysis'].get('experience', 'نامشخص')} سال")
                y -= 20
                c.drawString(50, y, f"مهارت‌ها: {', '.join(result['analysis'].get('skills', []))}")
                y -= 20
                c.drawString(50, y, f"تحصیلات: {result['analysis'].get('education', 'نامشخص')}")
                y -= 30
                c.drawString(50, y, "تحلیل:")
                y -= 20
                text_object = c.beginText(50, y)
                text_object.setFont('Vazir', 12)
                text_object.setLeading(14)
                for line in result['summary'].split('\n'):
                    text_object.textLine(line)
                    y -= 14
                c.drawText(text_object)
        
        c.showPage()
        c.save()
        output.seek(0)
        
        # حذف فایل موقت فونت
        os.remove(font_path)
        
        return send_file(
            output,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f'{filename}_analysis.pdf'
        )
    
    except Exception as e:
        logger.error(f"خطا در دانلود PDF: {str(e)}")
        return jsonify({'error': f'خطا در دانلود PDF: {str(e)}'}), 500

@app.route('/download/all/csv', methods=['GET'])
def download_all_csv():
    try:
        results = json.loads(request.args.get('results'))
        
        output = BytesIO()
        writer = csv.writer(output)
        writer.writerow(['Filename', 'Score', 'Experience', 'Skills', 'Education', 'Summary'])
        
        for result in results:
            if 'analysis' in result:
                writer.writerow([
                    result['filename'],
                    result['score'],
                    result['analysis'].get('experience', 'نامشخص'),
                    ', '.join(result['analysis'].get('skills', [])),
                    result['analysis'].get('education', 'نامشخص'),
                    result['summary']
                ])
        
        output.seek(0)
        return send_file(
            output,
            mimetype='text/csv',
            as_attachment=True,
            download_name='all_resumes_analysis.csv'
        )
    
    except Exception as e:
        logger.error(f"خطا در دانلود همه CSV: {str(e)}")
        return jsonify({'error': f'خطا در دانلود همه CSV: {str(e)}'}), 500

@app.route('/download/all/pdf', methods=['GET'])
def download_all_pdf():
    try:
        results = json.loads(request.args.get('results'))
        output = BytesIO()
        c = canvas.Canvas(output, pagesize=A4)
        
        # رجیستر فونت
        font_path = register_vazir_font()
        c.setFont('Vazir', 12)
        
        for result in results:
            if 'analysis' in result:
                y = 800
                c.drawString(50, y, f"رزومه: {result['filename']}")
                y -= 20
                c.drawString(50, y, f"امتیاز: {result['score']}")
                y -= 20
                c.drawString(50, y, f"تجربه: {result['analysis'].get('experience', 'نامشخص')} سال")
                y -= 20
                c.drawString(50, y, f"مهارت‌ها: {', '.join(result['analysis'].get('skills', []))}")
                y -= 20
                c.drawString(50, y, f"تحصیلات: {result['analysis'].get('education', 'نامشخص')}")
                y -= 30
                c.drawString(50, y, "تحلیل:")
                y -= 20
                text_object = c.beginText(50, y)
                text_object.setFont('Vazir', 12)
                text_object.setLeading(14)
                for line in result['summary'].split('\n'):
                    text_object.textLine(line)
                    y -= 14
                c.drawText(text_object)
                c.showPage()
        
        c.save()
        output.seek(0)
        
        # حذف فایل موقت فونت
        os.remove(font_path)
        
        return send_file(
            output,
            mimetype='application/pdf',
            as_attachment=True,
            download_name='all_resumes_analysis.pdf'
        )
    
    except Exception as e:
        logger.error(f"خطا در دانلود همه PDF: {str(e)}")
        return jsonify({'error': f'خطا در دانلود همه PDF: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
