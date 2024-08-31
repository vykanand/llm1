# Use the Miniconda base image from the Docker Hub
FROM continuumio/miniconda3

# Set the working directory
WORKDIR /app

# Copy environment.yml to the container
COPY environment.yml /app/

# Create and activate the Conda environment and install dependencies
RUN conda env create --file /app/environment.yml && \
    conda clean --all --yes

# Set the environment to be used for subsequent commands
SHELL ["conda", "run", "--name", "myenv", "/bin/bash", "-c"]

# Copy the rest of the application code
COPY app.py /app/

# Expose the port the app runs on
EXPOSE 5000

# Define the command to run the application
CMD ["python", "app.py"]
