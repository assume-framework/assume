# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

FROM python:3.12-slim

RUN useradd -m -s /bin/bash admin

RUN mkdir /src
WORKDIR /src
COPY README.md pyproject.toml .
# https://github.com/jazzband/pip-tools/pull/2221
RUN pip install "pip-tools<7.5.0"
RUN mkdir assume assume_cli
RUN touch assume/__init__.py
RUN pip-compile --resolver=backtracking -o requirements.txt ./pyproject.toml
RUN pip install --no-cache-dir -r requirements.txt

COPY README.md pyproject.toml /src
COPY assume /src/assume
COPY assume_cli /src/assume_cli
COPY examples /src/examples
ENV PATH=/home/admin/.local/bin:$PATH
RUN chown -R admin /src /home/admin
USER admin
RUN pip install -e .
ENV PYTHONUNBUFFERED=1
EXPOSE 9099
ENTRYPOINT ["assume"]
