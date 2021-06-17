import PyPDF2
from PyPDF2 import PdfFileReader

# Creating a pdf file object.
pdf = open("test.pdf", "rb")

# Creating pdf reader object.
pdf_reader = PyPDF2.PdfFileReader(pdf)

# Get the title of the file
title = pdf_reader.getDocumentInfo().title
 
# Checking total number of pages in a pdf file.
print("Total number of Pages:", pdf_reader.numPages)
print(f"Title: {title}")
 
# Creating a page object.
page = pdf_reader.getPage(1-34)
 
# Extract data from a specific page number.
print(page.extractText())
 
# Closing the object.
pdf.close()