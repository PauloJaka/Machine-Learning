import pandas as pd
import spacy
import re

def load_model(model_dir):
    """Carrega o modelo NER existente."""
    return spacy.load(model_dir)

def extract_entities(nlp, text):
    """Extrai entidades do texto usando o modelo NER."""
    doc = nlp(text)
    entities = {ent.label_: ent.text for ent in doc.ents}
    return entities

def extract_ram(title):
    """Extrai RAM usando regex, procurando pela palavra 'RAM' (case insensitive)."""
    match = re.search(r"(\d+)\s*GB\s*RAM", title, re.IGNORECASE)
    if match:
        return match.group(1) + " GB"
    return None

def extract_storage_capacity(title):
    """Extrai armazenamento usando regex, procurando valores seguidos de 'GB' e maiores que 16."""
    match = re.search(r"(\d+)\s*GB", title, re.IGNORECASE)
    if match and int(match.group(1)) > 16:
        return match.group(1) + " GB" 
    return None

def test_model(nlp, df):
    """Testa o modelo NER em um DataFrame e tenta extrair entidades de títulos."""
    results = []
    
    for index, row in df.iterrows():
        title = row['title']  # Supondo que a coluna do título se chama 'Título'
        attempts = 0
        entities_found = {}
        
        # Tenta extrair as entidades até 5 vezes
        while attempts < 5 and (len(entities_found) < 3):  # 3 entidades: Modelo, RAM, Armazenamento
            entities_found = extract_entities(nlp, title)
            attempts += 1

        # Se RAM não foi encontrado, tenta com regex
        if 'RAM' not in entities_found:
            ram_regex = extract_ram(title)
            if ram_regex:
                entities_found['RAM'] = ram_regex
        
        # Se Armazenamento não foi encontrado, tenta com regex
        if 'STORAGE' not in entities_found:
            storage_regex = extract_storage_capacity(title)
            if storage_regex:
                entities_found['STORAGE'] = storage_regex
        
        # Adiciona os resultados ao DataFrame
        results.append({
            'Título': title,
            'Modelo': entities_found.get('MODEL', None),
            'RAM': entities_found.get('RAM', None),
            'Armazenamento': entities_found.get('STORAGE', None),
        })

    return pd.DataFrame(results)

def main():
    """Função principal para testar o modelo e salvar os resultados em um CSV."""
    csv_file = "/home/paulo-jaka/Downloads/datasets_trabalho/tablets_new_training_data.csv"  # Atualize o caminho para o seu CSV
    model_dir = "/media/paulo-jaka/Extras/Machine-learning/modelo_ner_celulares_retrained"  # Diretório do seu modelo
    
    # Carrega o DataFrame do CSV
    df = pd.read_csv(csv_file)
    
    # Carrega o modelo NER
    nlp = load_model(model_dir)
    
    # Testa o modelo nos dados
    results_df = test_model(nlp, df)
    
    # Salva os resultados em um novo CSV
    results_df.to_csv("r2esultado_extração_entidades.csv", index=False)

if __name__ == "__main__":
    main()
