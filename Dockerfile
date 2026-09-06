# Read the doc: https://huggingface.co/docs/hub/spaces-sdks-docker

FROM python:3.9

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY --chown=user . /app

# main:app → our FastAPI app is in main.py (not app.py)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
