FROM python:3.10

RUN useradd -m -s /bin/bash admin

RUN mkdir /src
WORKDIR /src
COPY README.md LICENSE pyproject.toml .
#RUN python -m pip install --upgrade pip
# thats needed to use create the requirements.txt only 
RUN pip install pip-tools
RUN mkdir assume
RUN touch assume/__init__.py
RUN pip-compile --resolver=backtracking -o requirements.txt ./pyproject.toml
RUN pip install --no-cache-dir -r requirements.txt

RUN chown -R admin /src
RUN chown -R admin /home/admin

USER admin
COPY README.md pyproject.toml /src
COPY assume /src/assume
RUN pip install -e .

COPY examples /src/examples

CMD ["python", "-u" ,"./examples/example_01/example_01.py"]