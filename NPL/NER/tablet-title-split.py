import pandas as pd
import spacy
from spacy.util import minibatch, compounding
from spacy.training import Example
import random
import re

def load_data(file_path):
    return pd.read_csv(file_path)

def find_entity(entity_type, pattern, text, existing_entities):
    matches = list(re.finditer(pattern, text, re.IGNORECASE))
    for m in matches:
        start, end = m.start(), m.end()
        
        if not any(e_start < end and start < e_end for e_start, e_end, _ in existing_entities):
            return (start, end, entity_type)
    return None

def extract_entities(text, row):
    entities = []
    
    entity_patterns = {
        'MODEL': re.escape(row['Modelo']),
        'RAM': r'\b(\d+\s*GB RAM|\d+\s*G RAM|\d+\s*GB|\d+\s*RAM)\b',
        'STORAGE': r'\b(\d+\s*GB|\d+\s*TB)\b'
    }

    # Debugging prints
    print(f"Processing text: {text}")
    
    # Extract model
    model_entity = find_entity('MODEL', entity_patterns['MODEL'], text, entities)
    if model_entity:
        entities.append(model_entity)
    
    # Extract storage
    storage_values = re.findall(entity_patterns['STORAGE'], text, re.IGNORECASE)
    print(f"Storage values found: {storage_values}")  # Debugging print
    
    # Extract RAM
    ram_values = re.findall(entity_patterns['RAM'], text, re.IGNORECASE)
    print(f"RAM values found: {ram_values}")  # Debugging print
    
    if storage_values:
        storage_values.sort(key=lambda x: int(re.search(r'\d+', x).group()), reverse=True)
        storage_value = storage_values[0]
        storage_entity = find_entity('STORAGE', re.escape(storage_value), text, entities)
        if storage_entity:
            entities.append(storage_entity)
    
    if ram_values:
        ram_values.sort(key=lambda x: int(re.search(r'\d+', x).group()))
        ram_value = ram_values[0]
        ram_entity = find_entity('RAM', re.escape(ram_value), text, entities)
        if ram_entity:
            entities.append(ram_entity)
    else:
        potential_rams = re.findall(r'\b\d+\s*(?:RAM|G)\b', text, re.IGNORECASE)
        potential_rams = [x for x in potential_rams if int(re.search(r'\d+', x).group()) < 12]
        potential_rams.sort(key=lambda x: int(re.search(r'\d+', x).group()))
        
        if potential_rams:
            ram_value = potential_rams[0]
            ram_entity = find_entity('RAM', re.escape(ram_value), text, entities)
            if ram_entity:
                entities.append(ram_entity)
    
    return entities

def prepare_training_data(df):
    df['Título'] = df['Título'].astype(str).fillna('')
    df['Modelo'] = df['Modelo'].astype(str).fillna('')
    df['Armazenamento'] = df['Armazenamento'].astype(str).fillna('')
    training_data = []
    for _, row in df.iterrows():
        text = row['Título']
        entities = extract_entities(text, row)

        if entities:
            training_data.append((text, {"entities": entities}))
    
    return training_data

def train_model(training_data, iterations=100):
    nlp = spacy.blank("pt")
    ner = nlp.add_pipe("ner", last=True)
    
    # Add all entity labels
    for _, annotations in training_data:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.begin_training()
        for itn in range(iterations):
            random.shuffle(training_data)
            losses = {}
            batches = minibatch(training_data, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                for text, annotations in batch:
                    doc = nlp.make_doc(text)
                    example = Example.from_dict(doc, annotations)
                    nlp.update([example], drop=0.5, sgd=optimizer, losses=losses)
            print(f"Iteration {itn + 1}, Losses: {losses}")
    
    return nlp

def save_model(nlp, output_dir):
    nlp.to_disk(output_dir)

def main():
    csv_file = "/media/paulo-jaka/Extras/Machine-learning/arquivo_limpo.csv"
    output_dir = "modelo_ner_tablets"
    
    df = load_data(csv_file)
    training_data = prepare_training_data(df)
    
    # Print out some of the training data for debugging
    for text, annotations in training_data[:5]:  # Print first 5 for quick check
        print(f"Training data sample: {text} - {annotations}")

    nlp = train_model(training_data)
    save_model(nlp, output_dir)

if __name__ == "__main__":
    main()
