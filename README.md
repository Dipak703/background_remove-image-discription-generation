# background_remove-image-discription-generation
image background removal and image description generation using llama model
---

# AI-Powered Image Processing and Document Management Server

This project is an AI-driven FastAPI server that enables advanced image processing, automatic description generation, and efficient document management with vector search capabilities.

## Features

1. **Image Background Removal**:

   * Automatically remove the background of uploaded images.
   * Customize the background color with a hex code.

2. **AI-Based Description Generation**:

   * Generate detailed titles and descriptions for images, ideal for cataloging and e-commerce applications.

3. **Document Management**:

   * Upload and preprocess multiple documents.
   * Store documents in a vector database for semantic search and querying.

4. **Customizable Prompt**:

   * Modify system behavior by updating the AI’s prompt.

---

## Prerequisites

Ensure the following are installed:

* Python 3.9+
* Virtual environment (optional but recommended)

---

## Installation

### Step 1: Clone the Repository

```bash
git clone Dipak703/background_remove-image-discription-generation
cd server
```

### Step 2: Install Dependencies

Install all required Python packages using the provided `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Step 3: Configure Environment Variables

Create a `.env` file in the root directory and add your API keys and configuration:

```env
TOGETHER_API_KEY=<Your Together API Key>
```

---

## Usage

### Start the Server

Run the server using `uvicorn`:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The server will be accessible at `http://127.0.0.1:8000`.

---

### Endpoints Overview

#### 1. **Homepage**

* **URL**: `/`
* **Method**: GET
* **Description**: Welcome page for the API.

#### 2. **Remove Background**

* **URL**: `/remove-background`
* **Method**: POST
* **Parameters**:

  * `image`: Image file (optional if `imageUrl` is provided).
  * `imageUrl`: URL of the image.
  * `backgroundColor`: Hex code for the desired background color.
* **Response**: Returns the processed image with background removed.

#### 3. **Generate Description**

* **URL**: `/description_gen`
* **Method**: POST
* **Parameters**:

  * `image`: Image file (optional if `imageUrl` is provided).
  * `imageUrl`: URL of the image.
* **Response**: JSON containing the generated title and description.

#### 4. **Upload Documents**

* **URL**: `/upload_docs/`
* **Method**: POST
* **Parameters**: Multiple files as `UploadFile`.
* **Response**: Confirmation of successful uploads and document processing.

---

## Project Structure

```
.
├── templates/                # HTML templates for API responses
├── main.py                   # Core FastAPI server logic
├── requirements.txt          # List of dependencies
├── .env                      # Environment variables
├── README.md                 # Project documentation
└── ...
```

---

## Dependencies

Install the required Python packages using `pip`:

```bash
pip install -r requirements.txt
```

Main dependencies include:

* `fastapi`
* `uvicorn`
* `requests`
* `pillow`
* `transparent-background`
* `torch`
* `transformers`
* `chromadb`
* `langchain-ollama`
* `together`
* `python-dotenv`
* `numpy`
* `pydantic`

---

## Contributing

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-name`).
3. Commit changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature-name`).
5. Open a pull request.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for more information.

---

## Contact

For any questions or feedback, please reach out to \[[dipakgaddam102@gmail.com](mailto:dipakgaddam102@gmail.com)].

---

