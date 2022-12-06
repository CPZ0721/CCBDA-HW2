## CCBDA HW2 - Self-supervised Learning (SimCLR)
Brain Magnetic Resonance Imaging (MRI) data, which contains 7294 images of resolution 96x96 in 4 categories.
- unlabeled set: A full dataset containing 7294 images without labels.
- test set: A small subset containing 500 images with classification label.

## Dataset
- Unzip data.zip to `./`
    ```sh
    unzip data.zip 
    ```
- Folder structure
    ```
    ./
    ├── test/
    ├── unlabeled/ 
    ├── main.py   
    ├── model.py
    ├── requirements.txt
    ├── Trainset_Embedding.py
    ├── utils.py
    ```

## Environment
- Python 3.6 or later version
    ```sh
    pip install -r requirements.txt
    ```

## Train
```sh
python main.py
```
- Make Unlabeled Embedding
The prediction file is `emb_result.npy`.
