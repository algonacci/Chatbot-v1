# Chatbot-v1
The source code of Chatbot using Python and its library.

![alt_text](https://github.com/algonacci/Chatbot-v1/blob/main/thumbnail.jpg?raw=true)

# How to install and use it?
- Clone this repo to your local computer
- Create a Python virtual environment by typing this in the terminal:
```
python -m venv .venv
```

## Install and import library
- pip install tensorflow
- pip install keras
- pip install keras-models
- pip install pickle
- pip install nltk
- pip install flask

# Train the model and run it on a Flask web server
## 
- Prepare a bunch of Bot's Response data in JSON format
- Train the model with Bot's Response data in JSON by run the `training.py` file
- Edit some CSS styles and components in static folder
- Run **app.py** file, sometimes it needs to download some NLTK data also prepare the Tensorflow Library
- When it display localhost or http://127.0.0.1:5000/, just open it in the browser
- If it doesn't run the server, just type **flask run** in the terminal
- Start interact with the chatbot and test all the data that have been included in data.JSON
