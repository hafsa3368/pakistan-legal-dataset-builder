# Pakistan Legal Dataset Builder

A Python-based pipeline for collecting, processing, and organizing Pakistani legal judgments into a structured dataset for AI and Retrieval-Augmented Generation (RAG) applications.

## Project Overview

This project automates the collection and preprocessing of legal case documents from Pakistani court sources. It extracts metadata, performs OCR on scanned PDFs, converts judgments into structured JSON format, and prepares the data for downstream AI applications such as semantic search, vector databases, and Legal RAG systems.

## Features

* PDF scraping from Pakistani legal sources
* Automatic document downloading
* OCR support for scanned judgments using EasyOCR
* Text extraction from searchable PDFs
* Metadata extraction
* JSON generation for each legal document
* Excel metadata reports
* Logging and checkpoint support
* Modular Python scripts for different courts

## Project Structure

```text
.
├── supreme_court_scraper.py
├── shc_scraper.py
├── metadata.csv
├── supreme_court_ai_metadata.xlsx
├── extracted_text/
│   ├── *.json
├── logs/
├── checkpoints/
├── requirements.txt
└── README.md
```

## Technologies Used

* Python
* Selenium
* BeautifulSoup
* PyMuPDF
* EasyOCR
* Pillow
* NumPy
* OpenPyXL
* Pandas
* JSON

## Workflow

1. Scrape legal judgments.
2. Download PDF files.
3. Detect searchable and scanned PDFs.
4. Perform OCR when required.
5. Extract document text.
6. Generate structured metadata.
7. Save extracted text as JSON.
8. Export metadata to Excel.

## Output

The pipeline produces:

* Structured JSON files
* Metadata Excel sheets
* CSV reports
* Logs
* Organized legal dataset ready for AI applications

## Future Work

* ChromaDB integration
* FAISS vector indexing
* Neo4j GraphRAG
* Embedding generation
* Semantic legal search
* AI Legal Research Assistant
* Citation-aware Retrieval-Augmented Generation (RAG)

## Installation

Clone the repository:

```bash
git clone https://github.com/hafsa3368/pakistan-legal-dataset-builder.git
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the scraper:

```bash
python supreme_court_scraper.py
```

## Applications

* Legal AI
* Retrieval-Augmented Generation (RAG)
* Legal Information Retrieval
* NLP Research
* Court Judgment Analysis
* Legal Dataset Creation
* AI Research Projects

## Disclaimer

This repository is intended for educational and research purposes. Users should comply with the terms of service and copyright policies of the respective legal data sources.

## Author

**Hafsa Javaid**

MS Data Science Student

University of Management and Technology (UMT), Lahore
