# ::1ST STAGE - DEPENDENCIES::
FROM python:3.10-slim-buster AS build_depend

RUN apt-get update \
    && apt-get install -y \
         curl \
         build-essential \ 
         sqlite3 \
         graphviz

# Build Dev Environment
RUN pip install --upgrade pip

# Poetry installations
WORKDIR /app

COPY ./requirements.txt ./

RUN pip install -r requirements.txt

# Eventually re-build and add to requirements.txt
RUN apt-get install git -y
RUN pip install torchtext==0.15.1
RUN pip install py-stringsimjoin==0.3.3
RUN pip install vega==3.6.0

RUN rm /etc/localtime \
    && ln -s /usr/share/zoneinfo/US/Pacific /etc/localtime

# Clear update list
# RUN rm -rf /var/lib/apt/lists/*