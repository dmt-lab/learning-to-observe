'Download files from google drive, credit: https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url'

import requests
import argparse

FOLDS = {
    1 : '1BuXsF3c4cDfiU_k4IML3Ko36KAjLFyAI',
    2 : '1E6nCezE1PPS-lz6dsoo_RK_ehQ_DL62_',
    3 : '18N_WwFNbMQNFsLP40ldILQW3tI-dfW8V',
    4 : '11U1e0ojU7be1rR82tFpPJSZii36GA4gx',
    5 : '11W5n6bwuEvItpNL9ijjyIMN_TMBj0P3K'
}

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--all_folds', 
        help='Downloads 5 of the best classifier models, instead of just 1.',
        action='store_true'
        )
    return parser.parse_args()


def download_model(fold_id):
    model_dest = f'./data/pretrained_classifier_fold_{fold_id}.hdf5'
    print(f'Downloading model for fold {fold_id} to {model_dest}')
    download_file_from_google_drive(FOLDS[fold_id], model_dest)
    print(f'Downloaded model for fold {fold_id} to {model_dest}')


def download_aet():
    print(f'Downloading AET backbone')
    aet_id = '1zyi08pLjIMlrtV8ABcsCUdWBO2EHPmDE'
    aet_dest = './data/pretrained_aet_model.hdf5'
    download_file_from_google_drive(aet_id, aet_dest)
    print(f'AET backbone downloaded to {aet_dest}')



if __name__ == "__main__":

    args = parse_args()
    download_aet()

    if args.all_folds:
        for fold_id, gdrive_id in FOLDS.items():
            download_model(fold_id=fold_id)
    else:
        download_model(fold_id=5)