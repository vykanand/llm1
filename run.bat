#docker build -t my-flask-api .

RUN pip install --no-cache-dir --upgrade -r requirements.txt

my-flask-api

docker build --progress=plain -t my-flask-api .

docker run -d -p 5000:5000 my-flask-api

#conda create -n 38 python=3.8
#conda activate 38

pip freeze > requirements.txt

# Kaggle - !pip install --upgrade transformers datasets bitsandbytes scikit-learn textstat language_tool_python matplotlib peft pandas sentence-transformers torch
