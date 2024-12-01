import json


def create_captions_json(train_captions_file, val_captions_file, output_file):
    with open(train_captions_file, 'r') as f:
        train_data = json.load(f)

    with open(val_captions_file, 'r') as f:
        val_data = json.load(f)

    annotations = train_data['annotations'] + val_data['annotations']

    # Create vocabulary
    vocab = set()
    for annotation in annotations:
        caption = annotation['caption']
        vocab.update(caption.lower().split())

    # Create JSON structure
    data = {
        "annotations": annotations,
        "vocab": list(vocab)
    }

    with open(output_file, 'w') as f:
        json.dump(data, f)


# Specify paths to your files
train_captions_file = 'data/annotations/captions_train2017.json'
val_captions_file = 'data/annotations/captions_val2017.json'
output_file = 'data/captions.json'

create_captions_json(train_captions_file, val_captions_file, output_file)
