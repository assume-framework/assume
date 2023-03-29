FROM python:3.10

# Switch to root for install
USER root

COPY ./requirements.txt .
#RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

RUN useradd -s /bin/bash admin

RUN mkdir /src
RUN mkdir -p /home/admin
RUN chown -R admin /src
RUN chown -R admin /home/admin

USER admin
WORKDIR /src

COPY pyproject.toml README.md /src
COPY assume /src/assume


RUN pip install -e .

COPY examples /src/examples

CMD ["python", "-u" ,"./examples/example_01.py"]