import uvicorn
from fastapi import FastAPI
# import Model

# To run: `uvicorn app:ap --reload`

app = FastAPI()
# model = model

@app.get('/')
def index():
    '''
    Main Page
    '''
    return {'message': 'CancerMap'}

@app.get('/{name}')
def get_name(name: str):
    '''
    Test endpoint
    '''
    return {'message': f'{name}, upload an image to test! :)'}

@app.post('/predict')
def predict():
    ## ToDo: fill in predict
    # What will incoming data form be
    # What will we return
    return

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
