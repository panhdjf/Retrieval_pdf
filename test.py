# from PIL import Image
# def get_text_from_file(file):
#     _, file_extension = os.path.splitext(file.name.lower())
#     if file_extension == '.pdf':
#         return get_pdf_text([file])
#     elif file_extension in ('.jpg', '.jpeg'):
#         return get_text_from_image(file)
#     else:
#         st.warning(f"Unsupported file format: {file_extension}")
#         return ""

# def get_text_from_image(image_file):
#     image = Image.open(image_file)
#     text = pytesseract.image_to_string(image)
#     return text

# import pytesseract
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Đường dẫn đến tesseract.exe, hãy thay đổi cho phù hợp với máy của bạn
# def main():
#     # ...
#     pdf_docs = st.file_uploader(
#         "Upload your files here and click on 'Process'", accept_multiple_files=True, type=['pdf', 'jpg', 'jpeg'])
#     # ...


# def get_text_from_file(files):
#     text = ""
#     for file in files:
#         _, file_extension = os.path.splitext(file.name.lower())
#         if file_extension == '.pdf':
#             return text += get_pdf_text(file)
#         elif file_extension in ('.jpg', '.jpeg'):
#             return get_text_from_image(file)
#         else:
#             st.warning(f"Unsupported file format: {file_extension}")
#             return ""
from PIL import Image
import pytesseract
import os

image_file = 'module1.jpg'
image = Image.open(image_file)
text = pytesseract.image_to_string(image)
print(text)



# def get_text_from_file(files):
#     print("----------",files)
#     text = ""
#     for file in files:
#         _, file_extension = os.path.splitext(file.name.lower())
#         if file_extension == '.pdf':
#             print('0k-----')
#             text += get_pdf_text(file)
#         elif file_extension in ('.jpg', '.jpeg'):
#             print('************')
#             text += get_text_from_image(file)
#         else:
#             st.warning(f"Unsupported file format: {file_extension}")
#             return ""
#     return text


# def get_pdf_text(pdf_doc):
#     pdf_reader = PdfReader(pdf_doc)
#     for page in pdf_reader.pages:
#         text = page.extract_text()
#     return text

def get_text_from_image(image_file):
    image = Image.open(image_file)
    text = pytesseract.image_to_string(image)
    return text