FROM python:3.7


COPY . /app


WORKDIR /app


RUN pip install --upgrade pip


RUN pip install -r requirements.txt


#Expose the required port
EXPOSE 5000

#Run the command
CMD gunicorn app:app
