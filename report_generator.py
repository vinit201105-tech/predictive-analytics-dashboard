from fpdf import FPDF
import datetime

class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'Predictive Analytics Report', 0, 1, 'C')
        self.set_font('Arial', 'I', 10)
        self.cell(0, 10, f'Generated on: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def generate_pdf_report(dataset_summary, model_name, metrics, filename="predictive_analytics_report.pdf"):
    pdf = PDFReport()
    pdf.add_page()
    
    # Dataset Summary
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "1. Dataset Summary", 0, 1)
    pdf.set_font("Arial", '', 10)
    pdf.multi_cell(0, 10, dataset_summary)
    pdf.ln(5)
    
    # Model Used
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "2. Model Used", 0, 1)
    pdf.set_font("Arial", '', 10)
    pdf.cell(0, 10, f"Algorithm: {model_name}", 0, 1)
    pdf.ln(5)
    
    # Performance Metrics
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "3. Performance Metrics", 0, 1)
    pdf.set_font("Arial", '', 10)
    for k, v in metrics.items():
        pdf.cell(0, 10, f"{k}: {v:.4f}", 0, 1)
    pdf.ln(10)
    
    pdf.set_font("Arial", 'I', 10)
    pdf.multi_cell(0, 10, "Note: Visualizations can be downloaded directly from the interactive dashboard.")
    
    pdf.output(filename)
    return filename
