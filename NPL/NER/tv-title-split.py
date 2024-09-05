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

def detect_size_fallback(text):
    match = re.search(r'\b\d{2}\b', text)
    if match:
        return (match.start(), match.end(), 'SIZE')
    return None

def extract_entities(text, row):
    entities = []
    
    entity_patterns = {
        'MODEL': re.escape(row['modelo']),
        'SIZE': rf'\b{row["polegadas"]}\b[\s"]?(?:polegadas|")?',
        'RESOLUTION': re.escape(row['resolucao']),
        'TECHNOLOGY': re.escape(row['tecnologia'])
    }

    print(f"Processing text: {text}")
    
    # Extract model
    model_entity = find_entity('MODEL', entity_patterns['MODEL'], text, entities)
    if model_entity:
        entities.append(model_entity)
    
    # Extract size
    size_entity = find_entity('SIZE', entity_patterns['SIZE'], text, entities)
    if size_entity:
        entities.append(size_entity)
        print(f"Size entity found: {size_entity}")
    else:
        size_fallback_entity = detect_size_fallback(text)
        if size_fallback_entity:
            entities.append(size_fallback_entity)
            print(f"Size entity found using fallback: {size_fallback_entity}")
    
    # Extract resolution
    resolution_entity = find_entity('RESOLUTION', entity_patterns['RESOLUTION'], text, entities)
    if resolution_entity:
        entities.append(resolution_entity)
    
    # Extract technology
    technology_entity = find_entity('TECHNOLOGY', entity_patterns['TECHNOLOGY'], text, entities)
    if technology_entity:
        entities.append(technology_entity)
    
    return entities

def prepare_training_data(df):
    df['titulo'] = df['titulo'].astype(str).fillna('')
    df['modelo'] = df['modelo'].astype(str).fillna('')
    df['polegadas'] = df['polegadas'].astype(str).fillna('')
    df['tecnologia'] = df['tecnologia'].astype(str).fillna('')
    df['resolucao'] = df['resolucao'].astype(str).fillna('')
    
    training_data = []
    for _, row in df.iterrows():
        text = row['titulo']
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
    csv_file = "/media/paulo-jaka/Extras/Machine-learning/base-de-dados/training_data_tv2ajustado.csv"
    output_dir = "modelo_ner_tvs"
    
    df = load_data(csv_file)
    training_data = prepare_training_data(df)
    
    # Print out some of the training data for debugging
    for text, annotations in training_data[:5]:  # Print first 5 for quick check
        print(f"Training data sample: {text} - {annotations}")

    nlp = train_model(training_data)
    save_model(nlp, output_dir)

if __name__ == "__main__":
    main()
