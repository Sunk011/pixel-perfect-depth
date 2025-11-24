FROM pytorch/pytorch:2.5.1-cuda11.8-cudnn9-devel


# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first to leverage cache
COPY requirements.txt .

# Install python dependencies
# Upgrade pip first
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install gradio

# Copy the rest of the application
COPY . .

# Expose Gradio port
EXPOSE 7860

# Command to run the application
CMD ["python", "app.py"]
