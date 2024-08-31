# Use the Miniconda base image from the Docker Hub
FROM continuumio/miniconda3

# Set the working directory
WORKDIR /app

# Copy environment.yml and requirements.txt first
COPY environment.yml /app/
COPY requirements.txt /app/

# Create the Conda environment
RUN conda env create -f /app/environment.yml --verbose && \
    conda clean --all --yes

# Install Python dependencies
RUN ls -l /app/requirements.txt && pip install -r requirements.txt --verbose

# Copy the rest of the application code
COPY app.py /app/

# Expose the port the app runs on
EXPOSE 5000

# Define the command to run the application
CMD ["conda", "run", "--name", "myenv", "python", "app.py"]
