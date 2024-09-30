import pandas as pd
import spacy
from spacy.util import minibatch, compounding
from spacy.training import Example
import random
import re

def load_data(file_path):
    """Carrega os dados do CSV."""
    return pd.read_csv(file_path)

def load_model(model_dir):
    """Carrega o modelo NER existente."""
    return spacy.load(model_dir)

def find_entity(entity_type, pattern, text, existing_entities):
    """Encontra a entidade no texto com base no padrão fornecido."""
    matches = list(re.finditer(pattern, text, re.IGNORECASE))
    for m in matches:
        start, end = m.start(), m.end()
        # Verifica se a entidade não está sobrepondo com outras
        if not any(e_start < end and start < e_end for e_start, e_end, _ in existing_entities):
            return (start, end, entity_type)
    return None

def extract_entities_from_csv(text, row):
    """Extrai entidades APENAS com base nas informações do CSV."""
    entities = []
    
    # Verifica se existe valor no CSV para cada entidade
    if pd.notnull(row['Modelo']):
        model_entity = find_entity('MODEL', re.escape(row['Modelo']), text, entities)
        if model_entity:
            entities.append(model_entity)
    
    if pd.notnull(row['RAM']):
        ram_entity = find_entity('RAM', re.escape(row['RAM']), text, entities)
        if ram_entity:
            entities.append(ram_entity)
    
    if pd.notnull(row['Armazenamento']):
        storage_entity = find_entity('STORAGE', re.escape(row['Armazenamento']), text, entities)
        if storage_entity:
            entities.append(storage_entity)
    
    return entities

def prepare_training_data(df):
    """Prepara os dados de treinamento APENAS com base nos valores explícitos do CSV."""
    training_data = []
    for _, row in df.iterrows():
        text = row['Título']
        
        # Extrai entidades com base no CSV
        entities = extract_entities_from_csv(text, row)

        if entities:
            training_data.append((text, {"entities": entities}))
    
    return training_data

def train_model(nlp, training_data, iterations=200):
    """Treina o modelo NER com os dados de treinamento fornecidos."""
    ner = nlp.get_pipe("ner")
    
    # Adiciona os rótulos das entidades ao NER
    for _, annotations in training_data:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    # Treina o modelo
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.resume_training()
        for itn in range(iterations):
            random.shuffle(training_data)
            losses = {}
            batches = minibatch(training_data, size=compounding(8.0, 32.0, 1.01))
            for batch in batches:
                for text, annotations in batch:
                    doc = nlp.make_doc(text)
                    example = Example.from_dict(doc, annotations)
                    nlp.update([example], drop=0.3, sgd=optimizer, losses=losses)
            print(f"Iteration {itn + 1}, Losses: {losses}")

def save_model(nlp, output_dir):
    """Salva o modelo treinado no diretório especificado."""
    nlp.to_disk(output_dir)

def main():
    """Função principal para carregar os dados, treinar e salvar o modelo."""
    csv_file = "/media/paulo-jaka/Extras/Machine-learning/base-de-dados/tablet_training_data.csv"  # Caminho do CSV atualizado
    model_dir = "/media/paulo-jaka/Extras/Machine-learning/NPL/NER/trained_models/modelo_ner_tablets"  # Diretório do modelo existente
    output_dir = "modelo_ner_celulares_retrained"  # Diretório para salvar o modelo re-treinado
    
    # Carrega os dados do CSV
    df = load_data(csv_file)
    
    # Carrega o modelo existente
    nlp = load_model(model_dir)
    
    # Prepara os dados de treinamento
    training_data = prepare_training_data(df)
    
    # Treina o modelo com os dados de treinamento
    train_model(nlp, training_data)
    
    # Salva o modelo re-treinado
    save_model(nlp, output_dir)

if __name__ == "__main__":
    main()
