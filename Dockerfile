# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

FROM python:3.12-slim

RUN useradd -m -s /bin/bash admin

RUN mkdir /src
WORKDIR /src
COPY README.md pyproject.toml .
RUN pip install pip-tools
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
ARG PSEUDO_VERSION=0.1
RUN SETUPTOOLS_SCM_PRETEND_VERSION_FOR_ASSUME_FRAMEWORK=${PSEUDO_VERSION} pip install -e .
ENV PYTHONUNBUFFERED=1
EXPOSE 9099
ENTRYPOINT ["assume"]
