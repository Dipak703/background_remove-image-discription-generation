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
import numpy as np
from torch.quantization import quantize_dynamic
from transformers import CLIPProcessor, CLIPModel
from langchain_ollama import OllamaLLM

# Disable SSL verification and warnings
ssl._create_default_https_context = ssl._create_unverified_context
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

llm = OllamaLLM(model="llama2",base_url  = "http://localhost:11434",system="you are an jewellery expert",temperature=0.0)  
model=CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

quantized_model = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8) #8-bit quantized model

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

t=[j_type,p_gem,s_gem,design,size,metal,t_audience,visual_desc]

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
  res = llm.invoke(f"generate the description(2 to 4 lines) and title(3 to 5 words) of a object from the given features :{str(lst1)}")
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
    allow_origins=["*"],  # Change this to specific domains if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"status": "running", "message": "Background removal service is operational", "version": "1.0"}

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

        # Save to buffer
        output_buffer = io.BytesIO()
        output_image.save(output_buffer, format='PNG')
        output_buffer.seek(0)
        
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

        




if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)