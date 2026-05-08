# lightweight, official Python 3.10 Linux image
FROM python:3.10-slim

# create master folder inside the container called /app
WORKDIR /app

# copy lean requirements list into container
COPY api/requirements.txt .

# install all necessary Python libraries
RUN pip install --no-cache-dir -r requirements.txt

# copy local folders into the container's /app folder
COPY api/ /app/api/
COPY models/ /app/models/

# expose port 8080 for Google Cloud Run
EXPOSE 8080

# navigate to api folder and start server on port 8080
WORKDIR /app/api
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]