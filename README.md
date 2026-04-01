# Sementic_image_search

Retrieving images from the the Qdrant database by querying the database  in the form of text or image  

##  Project Overview

1.Storing images embeddings in the Qdrant database.

2.Query the database with text query or image query to retrieve the top k image results from the database.

3.OpenclipEmbedding is used to vectorise the both text and images while saving and retrieving data from the database.

4.Ingestion and Retrieval pipelines are decoupled

5.First ingest the images data into the qdrant database.

6.Then , query database either by text query or image query.

7.LLM is used to rewrite the text query written by the user.
## Project Structure
```
sementic-image-search                         
├─ data                                       
│  └─ query_images                            
│     └─ query.png                            
├─ images                                     
│  ├─ animal                                  
│  │  ├─ cat.jpeg                             
│  │  ├─ crocodile.jpeg                       
│  │  ├─ crocodile_1.png                      
│  │  ├─ dog.jpeg                             
│  │  ├─ elephant.jpeg                        
│  │  ├─ giraffe.webp                         
│  │  ├─ horse.webp                           
│  │  ├─ lion.jpeg                            
│  │  ├─ panda.jpg                            
│  │  ├─ tiger.jpeg                           
│  │  └─ zebra.jpeg                           
│  ├─ flower                                  
│  │  ├─ lavender.jpeg                        
│  │  ├─ lily.jpeg                            
│  │  ├─ lotus.jpg                            
│  │  ├─ marigold.jpeg                        
│  │  ├─ rose.jpg                             
│  │  ├─ sunflower.jpeg                       
│  │  └─ tulip.webp                           
│  ├─ furniture                               
│  │  └─ table.jpeg                           
│  ├─ general                                 
│  │  ├─ bottle.jpeg                          
│  │  ├─ car.webp                             
│  │  ├─ chair.jpeg                           
│  │  ├─ cycle.webp                           
│  │  ├─ laptop.jpeg                          
│  │  ├─ pen.webp                             
│  │  ├─ phone.jpeg                           
│  │  └─ table.jpeg                           
│  ├─ uncategorized                           
│  │  ├─ ak47.jpeg                            
│  │  ├─ crocodile_1.png                      
│  │  ├─ lion.jpeg                            
│  │  └─ sam_altman.jpeg                      
│  └─ weapon                                  
│     ├─ ak47.jpeg                            
│     ├─ crocodile_1.png                      
│     └─ pistol.jpeg                          
├─ logs                                       
│  ├─ 02_11_2026_02_39_00.log                 
│             
├─ semantic_image_search                      
│  ├─ backend                                 
│  │  ├─ exception                            
│  │  │  ├─ custom_exception.py               
│  │  │  └─ __init__.py                       
│  │  ├─ logger                               
│  │  │  ├─ custom_logger.py                  
│  │  │  └─ __init__.py                       
│  │  ├─ config.py                            
│  │  ├─ embeddings.py                        
│  │  ├─ ingestion.py                         
│  │  ├─ main.py                              
│  │  ├─ qdrant_client.py                     
│  │  ├─ query_translator.py                  
│  │  ├─ retriever.py                         
│  │  └─ __init__.py                          
│  ├─ notebooks                               
│  │  ├─ images                               
│  │  │  ├─ animal                            
│  │  │  │  ├─ cat.jpeg                       
│  │  │  │  ├─ crocodile.jpeg                 
│  │  │  │  ├─ crocodile_1.png                
│  │  │  │  ├─ dog.jpeg                       
│  │  │  │  ├─ elephant.jpeg                  
│  │  │  │  ├─ giraffe.webp                   
│  │  │  │  ├─ horse.webp                     
│  │  │  │  ├─ lion.jpeg                      
│  │  │  │  ├─ panda.jpg                      
│  │  │  │  ├─ tiger.jpeg                     
│  │  │  │  └─ zebra.jpeg                     
│  │  │  ├─ flower                            
│  │  │  │  ├─ lavender.jpeg                  
│  │  │  │  ├─ lily.jpeg                      
│  │  │  │  ├─ lotus.jpg                      
│  │  │  │  ├─ marigold.jpeg                  
│  │  │  │  ├─ rose.jpg                       
│  │  │  │  ├─ sunflower.jpeg                 
│  │  │  │  └─ tulip.webp                     
│  │  │  ├─ furniture                         
│  │  │  │  └─ table.jpeg                     
│  │  │  ├─ general                           
│  │  │  │  ├─ bottle.jpeg                    
│  │  │  │  ├─ car.webp                       
│  │  │  │  ├─ chair.jpeg                     
│  │  │  │  ├─ cycle.webp                     
│  │  │  │  ├─ laptop.jpeg                    
│  │  │  │  ├─ pen.webp                       
│  │  │  │  ├─ phone.jpeg                     
│  │  │  │  └─ table.jpeg                     
│  │  │  ├─ uncategorized                     
│  │  │  │  ├─ ak47.jpeg                      
│  │  │  │  ├─ crocodile_1.png                
│  │  │  │  ├─ lion.jpeg                      
│  │  │  │  └─ sam_altman.jpeg                
│  │  │  └─ weapon                            
│  │  │     ├─ ak47.jpeg                      
│  │  │     ├─ crocodile_1.png                
│  │  │     └─ pistol.jpeg                    
│  │  ├─ retrieved_results                    
│  │  │  └─ a39ea5c4f1794b0aa175202707c2a6bf  
│  │  │     ├─ result_0.png                   
│  │  │     ├─ result_1.png                   
│  │  │     └─ result_2.png                   
│  │               
│  └─ __init__.py                             
├─ ui                                         
│  └─ app.py                                  
├─ Dockerfile                                 
├─ project_structure.py                       
├─ pyproject.toml                             
├─ README.md                                  
├─ requirements.txt                           
└─ setup.py                                   
                         
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/suman520-git/sementic-image-search.git
cd sementic-image-search

# Create virtual environment
conda create -p venv python==3.12 -y
conda activate venv/ 

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuring variables

```bash
#For Qdrant database
QDRANT_API_KEY = "xxxx"  
QDRANT_URL = "xxxx"

#For Huggingface
HF_TOKEN = "xxxx"

#For open Ai model
OPENAI_API_KEY = "xxxx"

```

### 3. Images Data Ingestion to Database

```bash

# Run the command and ingest data to database through ingestion endpoint and keep serverup
step.1   uvicorn semantic_image_search.backend.main:app --reload 

```

### 4. Application  Usage

```bash
# For running the application through Sreamlit
step.1  streamlit run .\semantic_image_search\ui\app.py

```
## ML Application UI

# Search By text query
![image alt](https://github.com/suman520-git/sementic-image-search/blob/main/U1.png?raw=true)


# Search By Image query
![image alt](https://github.com/suman520-git/sementic-image-search/blob/main/U2.png?raw=true)

### 5.  Dockerization
```bash
# Build Docker Image
step.1 docker build -t image-search-app .

#Run Docker Container
step.2 docker run --rm -p 8000:8000 -p 8501:8501 image-search-app

```


