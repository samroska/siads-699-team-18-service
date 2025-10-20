# siads-699-team-18-service
Project for Fall '25 University of Michigan MADS Capstone Team 18

Team members:
- Samantha Roska
- Sawsan Allam
- Andre Luis Camarosano Onofre


## About the project
Skin cancer is a growing global health concern, with rising incidence rates and limited access to early diagnostic services in many regions- for example melanoma accounts for around 75% of skin cancer-related deaths (Didier et al., 2024). Timely detection is essential, as early-stage skin cancers are highly treatable, decreasing healthcare costs and improving health outcomes. Traditional diagnostic methods rely on specialist expertise and dermatoscopic imaging, which are often scarce in low-resource or remote settings. Advances in deep learning offer promising solutions by enabling automated classification of skin lesions from image data, potentially bridging these gaps. 

This motivated our proposal to develop and evaluate machine learning models that can distinguish cancerous vs. benign skin lesions, leveraging publicly available dermatoscopic and smartphone-based datasets. We propose two complementary models: one for healthcare providers, designed to analyze dermatoscopic images, and another for the general public, enabling classification of smartphone-captured images via a mobile-friendly web application inspired by the real-time mobile performance demonstrated by Oztel et al. (2023). This dual approach supports both professional diagnostics and early self-assessment, potentially prompting more timely medical consultations

## Data

PAD-UFES-20 is a smartphone-based dataset collected in primary care settings in Brazil. It is particularly suited for developing models for general public use through mobile interfaces.

## Project Architecture

The architecture of this project is a backend REST API service built with Python, designed to serve a machine learning model for skin lesion classification. The service exposes endpoints (such as /users) that accept image files via HTTP requests, process them using the BCN20000 model, and return predictions. It is intended to be consumed by a frontend application, enabling users to upload images and receive diagnostic results. The backend handles model loading, inference, and communication with the frontend.


## Local setup

1. Create A virtual python environment
    - `python -m venv venv `

2. activate the virtual environemnt 
    - `source venv/bin/activate`

3. (optional) change the CORS middleware if you're also using the frontend locally:
    - ` allow_origins=["http://localhost:5173"]`

4. run the start-up script
    - `sh start.sh`

5. curl the service
    - `curl -X POST -F "file=@path/to/your/file.jpg" http://localhost:8000/users`

6. stop the service
    - ` control + c`

7. change the CORS allow origins back to capstoneteam18.netlify.app before comitting any files
    - `allow_origins=["https://capstoneteam18.netlify.com"]`

Note: this project runs on port 8080. Unlike project siads-699-test-18-service.