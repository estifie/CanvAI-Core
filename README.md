# canvAI Core API

## Description

canvAI Core API is an AI image prediction API that is built with FastAPI. The AI is trained with the Google Quickdraw
dataset and is used to predict doodles.

## Dataset

The Quick Draw Dataset is a collection of 50 million drawings across 345 categories, contributed by players of the game
[Quick, Draw!](https://quickdraw.withgoogle.com). You can explore the dataset
[here](https://github.com/googlecreativelab/quickdraw-dataset)

## Installation

To install canvAI Core API, follow these steps:

1. Clone the repository

```bash
git clone https://github.com/axelnt/CanvAI-Core.git
```

2. Navigate to the project directory

```bash
cd CanvAI-Core
```

3. Install the dependencies

```bash
pip install poetry
```

4. Install the project dependencies

```bash
poetry install
```

5. Create `.env` file: You can create the `.env` file by copying the `.env.example` file.

```bash
cp .env.example .env
```

## Usage

### Training the Model:

Please note that the given model in the `models` folder can be used for development but for better results you'll need
to train it using the dataset.

1. **Get the dataset** You can get the dataset from the
   [Quick Draw Dataset](https://console.cloud.google.com/storage/browser/quickdraw_dataset/full/numpy_bitmap?pli=1).
   Download the datasets that you want (The datasets should be in `.npy` format.) and place it in the `data` directory.

2. **Load the dataset** After downloading the dataset, run the following command. This will load the dataset and create
   **features** and **labels** files for training. You can specify the location of the **features** and **labels** files
   from the environment variables `FEATURES_PATH` and `LABELS_PATH`.

```bash
poetry run load
```

3. **Train the model** After loading the dataset, you can train the model by running the following command. You can set
   the number of epochs and batch size for training if you want.

```bash
poetry run train --epoch 3 --batch-size 64
```

### Running the API

After training the model, you can run the API by running the following command:

```bash
poetry run start
```

The API's PORT is set to `8000` by default. You can change the port by setting the environment variable `PORT`.

## API Documentation

The default API is available at `http://localhost:8000/api/v1`. You can change the API version by setting the
environment variable `VERSION`.

The API documentation is available at `http://localhost:8000/docs` after running the API.

### Endpoints

1. **`POST /model/predict`** - This endpoint is used to predict the doodle. You can send the image data in the request
   body as a base64 encoded string. API will preprocess the image and let the model predict the doodle.

   #### Example Request

   ```json
   {
   	"image_data": "base64_encoded_image"
   }
   ```

   #### Example Response

   ```json
   {
   	"status": "success",
   	"data": {
   		"prediction": "airplane"
   	}
   }
   ```
