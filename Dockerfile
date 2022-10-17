FROM registry.access.redhat.com/ubi8/python-39

COPY . .

RUN python -m pip install .

ENTRYPOINT ["hello"]