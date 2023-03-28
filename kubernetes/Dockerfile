FROM pytorch/pytorch

WORKDIR "/workdir/"

COPY requirements.txt requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

RUN chown -R 1147:1003 .