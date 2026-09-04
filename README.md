# Epsilon Research Template 🔬

[![Use this template](https://img.shields.io/badge/Use%20this-Template-brightgreen?style=for-the-badge)](https://github.com/Epsilon-Data/epsilon-research-template/generate)

A ready-to-use template for privacy-preserving data research with the Epsilon SDK. Start analyzing sensitive datasets with automatic Python class generation and differential privacy protection.

## 🚀 Quick Start

1. **Click "Use this template"** above to create your research project
2. **Install the Epsilon SDK**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Start researching**:
   ```bash
   # Login to Epsilon platform
   epsilon login
   # See available datasets
   epsilon datasets
   # Download dataset example
   epsilon init <dataset_id>
   # Run 
   epsilon run
   # Build for deployment
   epsilon build
   # Change Server (you can use local server api ex : http://localhost:3334) 
   epsilon change-server <server_url>
   ```