FROM python:3.13

WORKDIR /app

# Create a directory for data
RUN mkdir -p /app/data

# download and extract prepared data
RUN bash -c "cd /app/data && wget https://kosmx.dev/iPz39eAjSFnifskPLQoTHpMt/bkk.zip && unzip bkk.zip"

# Install dependencies

# AMD version of PyTorch
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/rocm6.4
# NOVIDEO version of PyTorch
# RUN pip install --no-cache-dir torch torchvision

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# Copy source code and notebooks
#COPY src/ src/
#COPY notebook/ notebook/
#COPY run.sh run.sh


# Set the entrypoint to run the training script by default
# You can override this with `docker run ... python src/04-inference.py` etc.
# CMD ["/app/run.sh"]