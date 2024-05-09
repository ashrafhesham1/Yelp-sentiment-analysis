import requests
import os
import sys
    
def wrap_in_progress_bar(_iter):
    try:
        from tqdm import tqdm
        return tqdm(_iter)
    except:
        return _iter
    
    
def download_file_from_google_drive(_id, destination):
    
    print(f'Trying to fetch data to {destination}')
    
    def get_confirmation_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None
    
    def save_response_content(response, destination, CHUNK_SIZE=32768):
        with open(destination, 'wb') as f:
            for chunk in wrap_in_progress_bar(response.iter_content(CHUNK_SIZE)):
                if chunk:
                    f.write(chunk)
    
    BASE_URL = "https://docs.google.com/uc?export=download"
    
    session = requests.Session()
    
    response = session.get(BASE_URL, params={'id':_id}, stream=True)
    token = get_confirmation_token(response)
    
    if token:
        param = {'id':_id, 'confirm':token}
        response = session.get(BASE_URL, params=param, stream=True)
    
    save_response_content(response, destination)
    
    
    
if __name__ == "__main__":
    
    if len(sys.argv) > 2:
        print('Usage: python download.py [dataset_root_folder_name]')
        
    else:
        dataset_root = 'dataset'
        if len(sys.argv) == 2:
            dataset_root = sys.argv[1]
        
        os.makedirs(dataset_root, exist_ok=True)
        files_ids = {
        'raw_train.csv': '1xeUnqkhuzGGzZKThzPeXe2Vf6Uu_g_xM',
        'raw_test.csv': '1G42LXv72DrhK4QKJoFhabVL4IU6v2ZvB',
        }
        
        for name, _id in files_ids.items():
            file_path = os.path.join(dataset_root, name)
            if os.path.exists(file_path):
                continue
            download_file_from_google_drive(_id, file_path)        