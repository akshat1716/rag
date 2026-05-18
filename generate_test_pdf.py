from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

def generate_test_pdf(filename):
    c = canvas.Canvas(filename, pagesize=letter)
    c.drawString(100, 750, "Project RAG Test Document")
    c.drawString(100, 730, "-------------------------")
    c.drawString(100, 700, "This is a controlled test document to verify the retrieval system.")
    c.drawString(100, 650, "The secret password for the system is: BlueberryPancakes.")
    c.drawString(100, 600, "If you can read this, the ingestion pipeline is working correctly.")
    c.save()
    print(f"Created {filename}")

if __name__ == "__main__":
    generate_test_pdf("data/pdf/test_document.pdf")
