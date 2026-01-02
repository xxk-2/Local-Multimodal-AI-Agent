import os
import shutil
from pypdf import PdfReader

def extract_text_from_pdf(pdf_path, max_pages=2):
    """
    从 PDF 提取文本。为了效率，默认只提取前 2 页（通常包含摘要和引言）。
    """
    try:
        reader = PdfReader(pdf_path)
        text = ""
        # 限制读取页数，避免处理整本书耗时过长，且前几页通常包含分类所需的核心信息
        for i, page in enumerate(reader.pages):
            if i >= max_pages: 
                break
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        print(f"Error reading PDF {pdf_path}: {e}")
        return ""

def move_file(src_path, dest_folder):
    """
    移动文件到目标文件夹，如果文件夹不存在则创建。
    """
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    
    filename = os.path.basename(src_path)
    dest_path = os.path.join(dest_folder, filename)
    shutil.move(src_path, dest_path)
    return dest_path