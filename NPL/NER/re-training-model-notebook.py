import pandas as pd
import spacy
from spacy.util import minibatch, compounding
from spacy.training import Example
import random
import re

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def prepare_training_data(df):
    training_data = []
    for _, row in df.iterrows():
        text = row['title']
        entities = []

        def find_entity(entity_type, pattern, existing_entities):
            match = re.finditer(pattern, text, re.IGNORECASE)
            for m in match:
                start, end = m.start(), m.end()
                if not any(e_start < end and start < e_end for e_start, e_end, _ in existing_entities):
                    return (start, end, entity_type)
            return None

        entity_types = {
            'MODEL': row['model'],
            'CPU': row['cpu'],
            'GPU': row['gpu'],
            'RAM': row['ram'],
            'SSD': row['ssd']
        }

        for entity_type, value in entity_types.items():
            if pd.notna(value):
                entity = find_entity(entity_type, re.escape(str(value)), entities)
                if entity:
                    entities.append(entity)

        if entities:
            training_data.append((text, {"entities": entities}))

    return training_data

def train_model(training_data, model_path, iterations=350):
    nlp = spacy.load(model_path)  # Carrega o modelo pré-treinado
    ner = nlp.get_pipe("ner")

    for _, annotations in training_data:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])  # Adiciona novas etiquetas de entidades

    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.resume_training()  # Retoma o treinamento
        for itn in range(iterations):
            random.shuffle(training_data)
            losses = {}
            batches = minibatch(training_data, size=compounding(8.0, 64.0, 1.01))

            for batch in batches:
                for text, annotations in batch:
                    doc = nlp.make_doc(text)
                    example = Example.from_dict(doc, annotations)
                    nlp.update([example], drop=0.3, sgd=optimizer, losses=losses)
            print(f"Iteration {itn + 1}, Losses: {losses}")

    return nlp

def save_model(nlp, output_dir):
    nlp.to_disk(output_dir)

def main():
    csv_file = "/media/paulo-jaka/Extras/Machine-learning/base-de-dados/notebook_new_samples_att.csv"
    output_dir = "modelo_ner_notebooks4_last"
    model_path = "/media/paulo-jaka/Extras/Machine-learning/NPL/NER/trained_models/modelo_ner_notebooks"  # Caminho para o modelo pré-treinado

    df = load_data(csv_file)
    training_data = prepare_training_data(df)

    nlp = train_model(training_data, model_path)
    save_model(nlp, output_dir)

if __name__ == "__main__":
    main()
