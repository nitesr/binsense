FROM python:3.11.8

WORKDIR /build
COPY ./requirements.txt .
COPY ./apps/binfinder/app-requirements.txt .
COPY ./libs ./libs
RUN ls -1
RUN pip3 install -r requirements.txt
RUN pip3 install ./libs
RUN pip3 install -r app-requirements.txt

RUN rm -fR /build

WORKDIR /work
COPY ./apps/binfinder/app .
COPY ./apps/binfinder/reactapp/dist/index.html ./templates
COPY ./apps/binfinder/reactapp/dist/assets/* ./assets


EXPOSE 8080
CMD ["uvicorn", "fastapi_app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "2"]