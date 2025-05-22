
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import requests
from PIL import Image
import io
import warnings
from transparent_background import Remover
import ssl
import torch
import re
import json
import os
from dotenv import load_dotenv
import numpy as np
from torch.quantization import quantize_dynamic
from transformers import CLIPProcessor, CLIPModel
from langchain_ollama import OllamaLLM
from typing import Annotated
from fastapi import FastAPI, Request
# from fastapi.templating import Jinja2Templates
from fastapi.responses import FileResponse, StreamingResponse
import os
from llama_index.core import SimpleDirectoryReader
import chromadb
from fastapi import File, UploadFile
from fastapi.templating import Jinja2Templates
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from together import Together
import together

load_dotenv()
together.api_key = os.getenv("TOGETHER_API_KEY")

templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))

#setup the system prompt for model
system_prompt="""You are GemCapture AI, the official chatbot for GemCapture, a product developed by DiaLumin — a leader in technology-driven industrial manufacturing solutions.

Behavior Rules:
Answer ONLY questions about GemCapture.
Respond in a single paragraph, maximum 3 sentences.
Never split into two paragraphs.
If asked something unrelated to GemCapture, reply : I'm only able to assist with GemCapture-related questions. For other inquiries, please visit our support page or call us at +91 98988 33644.
If you do not know the answer, reply : I'm unable to answer that. Please contact our team at +91 98988 33644.
Always end every response with : For further help, feel free to call us at +91 98988 33644.

About DiaLumin (Context Only) : 
Mission: Provide innovative, technology-driven industrial manufacturing solutions that enhance efficiency, precision, and scalability​.
Vision: To become a global leader in smart, technology-driven industrial manufacturing​.
Values: Innovation, excellence, efficiency, customer satisfaction​.

Services:
CAD Designing (precise and innovative product designs)​,PCB Designing (high-quality electronic designs)​,Software Development (custom solutions)​,Web Development (user-friendly websites and apps)​,AI/ML Solutions (advanced AI models for optimization and decision-making)​.

Why DiaLumin:
Innovative solutions, expertise and precision, customer-centric approach, future-ready technologies, reliable support.

Approved Knowledge Base about GemCapture:
GemCapture is an advanced imaging solution developed by DiaLumin, offering HD photo capture, 360-degree video creation, and professional product imaging technologies.
GemCapture Features : HD imaging, 360° rotating stage and top hangers, background removal, photo editing tools, secure cloud storage, multi-device access.
Supported Formats : JPG, PNG, TIFF, MP4, GIF.
System Requirements : Windows, macOS, updated Chrome/Edge/Safari browsers.
Support Methods : Email, live chat, phone.

"""
# system_prompt="""
# System Prompt for GemCapture Chatbot
# You are GemCapture AI, a smart and professional chatbot designed to assist users with questions related to GemCapture. Your role is to provide clear, concise, and helpful responses exclusively about GemCapture's products, features, and services.

# Guidelines:
# Strict Scope: Only answer questions related to GemCapture. If a question is unrelated, politely redirect the user to GemCapture’s official resources.
# Product Knowledge: Provide accurate and up-to-date information about:
# {HD image and 360-degree video capture,
# Rotating stage and top hangers for 360-degree product photography,
# (Photo editing, background removal, and cloud storage),
# Multi-device access and real-time syncing,
# Any other relevant GemCapture features}
# Clarity & Conciseness: Keep responses short, direct, and informative, elaborating only when necessary.
# Brand Consistency: Maintain a professional, friendly, and helpful tone that aligns with GemCapture’s brand identity.
# Proactive Assistance: Suggest relevant features, troubleshooting steps, or how users can get the most out of GemCapture.
# No Speculation: If a question is beyond your knowledge, respond with:
# "I'm here to assist with GemCapture-related questions. For other inquiries, please visit our official support page or contact our team."
# Company-Specific FAQs for GemCapture AI

# What is GemCapture?
# GemCapture is an advanced imaging solutio




# How does GemCapture's 360-degree capture work?
# GemCapture uses a motorized rotating stage and top hangers that rotate 360 degrees, ensuring seamless and consistent imaging from all angles.

# Can I remove backgrounds from my photos?
# Yes! GemCapture includes an advanced background removal tool that allows you to create clean, professional-looking images effortlessly.

# Is my content securely stored?
# Absolutely! All images and videos are stored in our secure cloud storage, allowing easy access from multiple devices with real-time syncing.

# What file formats does GemCapture support?
# GemCapture supports various formats, including JPG, PNG, TIFF, MP4, and GIF, ensuring flexibility for your workflow.

# Can I access GemCapture from multiple devices?
# Yes! GemCapture offers multi-device real-time access, so you can work seamlessly across your desktop, tablet, or mobile device.

# What are the system requirements for GemCapture?
# GemCapture is compatible with Windows, macOS, and modern web browsers. For the best experience, we recommend using an updated browser like Chrome, Edge, or Safari.

# Do you offer customer support?
# Yes! Our support team is available via email, live chat, and phone. You can also visit our Help Center for guides and FAQs.

# Can I share my captured images and videos?
# Yes! With our built-in cloud sharing, you can easily generate shareable links or download your content for distribution.

# How do I sign up for GemCapture?
# You can sign up directly on our website and choose a plan that fits your needs. We offer both free trials and premium plans with enhanced features.

# Unique Chatbot Behaviors for GemCapture AI
# Proactive Assistance:

# If a user asks about features, the bot can suggest related ones (e.g., if they ask about 360-degree capture, it can mention background removal and cloud storage).
# Troubleshooting Help:

# If a user reports an issue (e.g., "My image isn’t loading"), the bot can guide them through basic troubleshooting steps.
# Example: "Try refreshing your browser or clearing your cache. If the issue persists, please contact our support team!"
# Instant Pricing Info & Subscription Guidance:

# If a user asks about pricing, the bot can provide an overview of available plans and direct them to the pricing page.
# Product Demo Invitation:

# If a user seems interested but unsure, the bot can offer a demo or link to a video tutorial.
# Example: "Would you like to watch a quick demo on how GemCapture works? Click here!"
# Lead Capture for Support & Sales:

# If a user has a complex question, the bot can collect their email and forward the inquiry to the support team.
# Example: "I’d be happy to connect you with a specialist! Please share your email, and our team will reach out shortly."

# """
# Disable SSL verification and warnings
ssl._create_default_https_context = ssl._create_unverified_context
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

llm0 = OllamaLLM(model="llama2",base_url  = "http://localhost:11434",system="you are an jewellery expert",temperature=0.0) 
llm1 = OllamaLLM(model="llama2",base_url  = "http://localhost:11434",system=system_prompt,role="assistant",temperature=0.0,max_token=500)  
# llm1 = OllamaLLM(
#     model="llama2",
#     temperature=0.5,
#     system="""
# You are a helpful assistant for GemCapture — a professional photography and video capture solution designed for jewelry retailers. 
# GemCapture captures HD images and 360-degree videos with uniform lighting, using a rotating stage and top hangers. 
# It includes features like background removal, image editing, secure cloud storage, and real-time syncing across devices. 
# Respond with accurate information about GemCapture's features and avoid confusion with unrelated software tools.
# """,
#     max_token=5
# )


model=CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
chroma_client = chromadb.PersistentClient(path='apidata')
data_collection = chroma_client.get_or_create_collection(name="company_data")
quantized_model = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8) #8-bit quantized model

# # j_type=["necklace", "finger ring","single earring","earrings","necklace without chain","bangels","pendant without chain"]
# j_type=["necklace", "ring","bracelet",  "finger ring","single earring","earrings","necklace without chain","bangels","pendant without chain", "brooch", "pendant", "hairclip", "anklet"]
# p_gem=["diamond center stone", "ruby center stone", "emerald center stone", "sapphire center stone", "amethyst center stone", "pearl center stone", "topaz center stone", "opal center stone", "garnet center stone", "aquamarine center stone"]
# s_gem=["surrounded by small diamond","surounded by nothing or no secondary stone"]
# # design=[ "modern design", "classic design", "minimalist design", "flower design","round shaped", "oval shaped", "square shaped", "cushion shaped", "pear shaped"]
# # design=[ "modern design", "classic design", "minimalist design", "flower design"]
# shape_centre_gem = ["round shaped", "oval shaped", "square shaped", "cushion shaped", "pear shaped"]
# style = ["Timeless", "Classic", "Chic", "Modern", "Vintage", "Contemporary", "Bohemian", "Minimalist", "Baroque", "Artistic"]
# appearance = ["elegant", "sparkling", "dazzling", "gleaming", "shiny", "glittering", "radiant", "polished", "glossy", "luminous", "matt"]
# ity = ["exquisite", "luxurious", "premium", "refined", "Impeccable", "Flawless", "Intricate", "Pristine", "Delicate", "Sophisticated"]

# # size=["small size", "medium size", "large size"]
# metal=["gold", "silver"]
# # occasion=["wedding occasion", "casual occasion", "formal occasion", "party occasion", "gifting ", "travel"]
# # t_audience=["women", "men", "teen", "fashionista", "casual"]
# t_audience=["women", "men"]
# visual_desc=["dazzling", "radiant", "glittering", "shimmering", "captivating", "bold", "playful", "charming"]
j_type=["necklace", "finger ring","single earring","earrings","necklace without chain","bangels","pendant without chain"]
p_gem=["diamond center stone", "ruby center stone", "emerald center stone", "sapphire center stone", "amethyst center stone", "pearl center stone", "topaz center stone", "opal center stone", "garnet center stone", "aquamarine center stone"]
s_gem=["surrounded by small diamond","surounded by nothing or no secondary stone"]
design=[ "modern design", "classic design", "minimalist design", "flower design","round shaped", "oval shaped", "square shaped", "cushion shaped", "pear shaped"]
size=["small size", "medium size", "large size"]
metal=["gold", "silver"]
# occasion=["wedding occasion", "casual occasion", "formal occasion", "party occasion", "gifting ", "travel"]
# t_audience=["women", "men", "teen", "fashionista", "casual"]
t_audience=["women", "men"]
visual_desc=["dazzling", "radiant", "glittering", "shimmering", "captivating", "bold", "playful", "charming"]


t=[j_type,p_gem,s_gem,size,metal,t_audience,visual_desc,design]

app = FastAPI()
def generating_prompt(image):
  lst1=[]
  image=image
   #add the path of image to generate description
  for items in t:
    inputs = processor(text=items, images=image, return_tensors="pt", padding=True)
    # with torch.cuda.amp.autocast():
    outputs = quantized_model(**inputs)
    # print(outputs)
    logits_per_image = outputs.logits_per_image  # this is the image-text similarity score 
    probs = logits_per_image.softmax(dim=1).detach().numpy()
    probs=np.array(probs)
    # print(probs)
    indes=np.argmax(probs)
    lst1.append(items[indes])
  res = llm0.invoke(f"generate the description(2 to 4 lines) and title(3 to 5 words) of a object from the given features :{str(lst1)}")
  text = res
  substring = "Title:"
  desc="Description:"
  match0 = re.search(substring, text)
  match1 = re.search(desc,text)
  if match0 and match1:
    title=text[match0.start():match1.start()]
    description = text[match1.start():]
    X = title.split(":")
    y = description.split(":")
    di = {X[0]:X[1],y[0]:y[1]}
    json_object = json.dumps(di)
    return json_object
  else:
    return f"The substring '{substring}' is not found."
# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow requests only from specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



TOGETHER_API_KEY="29ef1ee07e9f9ddcadf34c8289736b967c94afc6b95bfc12adf81bd7205bea32"

client = Together() # auth defaults to os.environ.get("TOGETHER_API_KEY")


@app.get('/')
async def index(request:Request):
    return templates.TemplateResponse("index.html", {"request": request})
@app.post("/remove-background")
async def remove_background(
    request: Request,
    image: UploadFile = File(None), 
    imageUrl: str = Form(None), 
    backgroundColor: str = Form(None)):
    try:
        input_image = None
        
        # Handle JSON request
        if request.headers.get("content-type") == "application/json":
            data = await request.json()
            imageUrl = data.get("imageUrl")
            backgroundColor = data.get("backgroundColor")
        
        if image:
            # Handle direct image upload
            input_image = Image.open(io.BytesIO(await image.read()))
        elif imageUrl:
            # Handle image URL
            response = requests.get(imageUrl)
            if response.status_code != 200:
                raise HTTPException(status_code=400, detail="Failed to fetch image from URL")
            input_image = Image.open(io.BytesIO(response.content))
        else:
            raise HTTPException(status_code=400, detail="No image or image URL provided")
        
        # Initialize remover
        remover = Remover()
        
        # Convert input_image to RGB mode
        input_image = input_image.convert('RGB')
        
        # Remove background using new method
        output_image = remover.process(input_image, type='rgba'
                                       
                                       )
        
        # If background color is specified, apply it
        if backgroundColor:
            # Convert hex to RGB
            bg_color = tuple(int(backgroundColor.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
            
            # Create new image with background color
            background = Image.new('RGBA', output_image.size, bg_color + (255,))
            # Use alpha channel as mask
            background.paste(output_image, (0, 0), output_image)
            output_image = background
        try:
            input_image.save(f"D:/server/remover_image/{image.filename}.png")
        except: 
            pass

        # Save to buffer
        output_buffer = io.BytesIO()
        output_image.save(output_buffer, format='PNG')
        output_buffer.seek(0)
        # with open(image.filename, "wb") as f:
        #     f.write(output_image)
        
        return StreamingResponse(output_buffer, media_type="image/png", headers={"Content-Disposition": "attachment; filename=removed_bg.png"})
    
    except Exception as e:
        print(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/description_gen")
async def description_gen(
    request: Request,
    image: UploadFile = File(None), 
    imageUrl: str = Form(None) ):
    try:
        input_image = None
        
        # Handle JSON request
        if request.headers.get("content-type") == "application/json":
            data = await request.json()
            imageUrl = data.get("imageUrl")
        
        if image:
            # Handle direct image upload
            input_image = Image.open(io.BytesIO(await image.read()))
        elif imageUrl:
            # Handle image URL
            response = requests.get(imageUrl)
            if response.status_code != 200:      
                raise HTTPException(status_code=400, detail="Failed to fetch image from URL")
            input_image = Image.open(io.BytesIO(response.content))
        else:
            raise HTTPException(status_code=400, detail="No image or image URL provided")
        
    
        
        # Convert input_image to RGB mode
        input_image = input_image.convert('RGB')
        output = generating_prompt(input_image)

        return StreamingResponse(output, media_type="text/json", headers={"Content-Disposition": "attachment; filename=discription.json"})
    
    except Exception as e:
        print(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload_docs/")
async def create_upload_files(files: Annotated[list[UploadFile], File(description="Multiple files as UploadFile")],):
    for file in files:
        data = await file.read()
        with open(file.filename, "wb") as f:
            f.write(data)
        os.replace(f"D:\\server\\{file.filename}",f"D:\\server\\data\\{file.filename}")
    candidate_data = SimpleDirectoryReader("D:/server/data").load_data()
    #data preprocessing
    documents = []
    metadata = []
    ids = []
    for items in candidate_data:
        items=dict(items)
        print(type(items))
        print(items)
        documents.append(items["text_resource"].text)
        metadata.append(items["metadata"])
        ids.append(items["id_"])
    #pushing the resume data to vector database
    data_collection.upsert(documents=documents,metadatas=metadata,ids=ids)
    return {"filenames": [file.filename for file in files]}


'''@app.post('/output')
async def get_prompt(request: PromptRequest):  # Expecting JSON body
    prompt = request.prompt  # Extract the prompt
    # candidate_result = data_collection.query(query_texts=[prompt], n_results=2)
    new_prompt = prompt 
    # + str(candidate_result["documents"][0])

    return StreamingResponse(llm1.invoke(new_prompt), media_type="text/json", 
                             headers={"Content-Disposition": "attachment; filename=description.json"})


'''
class PromptRequest(BaseModel):
    prompt: str
@app.post('/output')
async def get_prompt(request: PromptRequest):  # Expecting JSON body
    prompt = request.prompt  # Extract the prompt
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        messages=[{"role": "assistant", "content": "context : "+system_prompt+"/n"+"user question:"+prompt}],
        # system = system_prompt,
        temperature=0.0
        )

    return StreamingResponse(response.choices[0].message.content, media_type="text/json", 
                             headers={"Content-Disposition": "attachment; filename=description.json"})


if __name__ == "__main__":
    uvicorn.run(app, host="192.168.1.50", port=6565)
     


