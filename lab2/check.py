from langchain_community.document_loaders import PyPDFLoader

# เปลี่ยนเป็นชื่อไฟล์ของคุณ
loader = PyPDFLoader("67008_6.pdf") 
docs = loader.load()

print("--- Testing Page 1 Content ---")
print(docs[0].page_content[:50]) # ปริ้น 500 ตัวอักษรแรก