import urllib.request
import shutil
import gzip
import json
import re
import numpy as np
from unidecode import unidecode
from pathlib import Path
from collections import defaultdict

from tqdm import tqdm
from dblp_parser import parse_dblp, parse_dblp_person

URL = 'http://dblp.org/xml/'
DATA_PATH = Path('data', 'bibliometric_dataset')
VERBOSE = 1
S2_API_KEY = 'L5FJHxOv2y4y0SLYWIEMW3iNJO7o6bOcdVi7f5c5'
SEPARATION_YEAR = 2016


def download_dblp() -> None:
    source_gz = f'{URL}dblp.xml.gz'
    source_dtd = f'{URL}dblp.dtd'
    target_gz = Path(DATA_PATH, 'dblp.xml.gz')
    target_dtd = Path(DATA_PATH, 'dblp.dtd')

    print('   Downloading file ' + source_gz)
    with urllib.request.urlopen(source_gz) as response, open(target_gz, 'wb') as fh:
        shutil.copyfileobj(response, fh)
    print('   Downloading file ' + source_dtd)
    with urllib.request.urlopen(source_dtd) as response, open(target_dtd, 'wb') as fh:
        shutil.copyfileobj(response, fh)
    print('   Download finish!')
    print()


def unzip_dblp() -> None:
    source = Path(DATA_PATH, 'dblp.xml.gz')
    target = Path(DATA_PATH, 'dblp.xml')

    with gzip.open(source, 'rb') as f_in:
        with open(target, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    print()


def extract_publications():
    source = Path(DATA_PATH, 'dblp.xml')
    target = Path(DATA_PATH, 'dblp.json')

    parse_dblp(source, target)
    print()


def extract_ai_publications() -> list:
    source = Path(DATA_PATH, 'dblp.json')
    source_venues = Path(DATA_PATH, 'ai_venues.json')
    target_pubs = Path(DATA_PATH, 'ai_dblp.jsonl')
    authors = set()
    with open(source_venues, "r", encoding="utf-8") as f:
        tmp = json.load(f)
    # Create a dict for all instances
    venues = dict(pair for d in tmp.values() for pair in d.items())
    venues_set = set()
    for k, v in venues.items():
        venues_set.add(k)
        venues_set.update(v)

    def get_disambiguated_venue(venue_name: str):
        if venue_name in venues:
            return venue_name
        else:
            for k, v in venues.items():
                if venue_name in v:
                    return k

    print(f'\tParsing {source}')
    with open(target_pubs, "w", encoding="utf-8") as out_f:
        with open(source, "r", encoding="utf-8") as in_f:
            for line in tqdm(in_f):
                line = json.loads(line)
                if line['booktitle']:
                    curr_venue = line['booktitle'][0]
                elif line['journal']:
                    curr_venue = line['journal'][0]
                else:
                    print("ERROR: No booktitle or journal: ", line)
                    continue
                curr_venue = re.sub(" \([0-9]+\)$", "", curr_venue)
                if curr_venue in venues_set:
                    line['venue'] = get_disambiguated_venue(curr_venue)
                    json.dump(line, out_f)
                    out_f.write("\n")
                    authors.update(line['author'])
    print('   Parse finish! File ai_dblp.jsonl created!')
    print()
    return list(authors)

def match_semantic_scholar_papers():
    source = Path(DATA_PATH, 'ai_dblp.jsonl')
    semantic_scholar_path = Path(DATA_PATH, 'semantic_scholar')
    target = Path(DATA_PATH, 'ai_matched.jsonl')
    
    def de_list(x, parse_int: bool = False):
        if isinstance(x, list):
            if parse_int:
                return int(x[0])
            return x[0]
        if parse_int:
            return int(x)
        return x

    def get_doi(line):
        """
        Get doi for a given line of the data, useful for semantic_scholar matching"
        """
        if "ee" in line and len(line['ee']) != 0:
            for x in de_list(line["ee"]):
                if "doi" in x:
                    return x.replace("https://doi.org/", "")

    pubs = []
    with open(source, "r", encoding="utf-8") as f:
        for line in f:
            pubs.append(json.loads(line))
    removed_indices = set()
    print("\tExtracting titles ...")
    titles = defaultdict(list)
    [titles[x['title'].strip(".").lower()].append(i) for i, x in enumerate(pubs)]
    files = [str(file_path) for file_path in Path.iterdir(semantic_scholar_path) if "papers-part" in str(file_path)]
    counter = 1
    not_matched = 0
    with open(target, 'w', encoding="utf-8") as out_f:
        for file_path in files:
            print()
            print("\tReading file ... (", str(counter), "/", str(len(files)), ")")
            with gzip.open(Path(file_path), 'rb') as in_f:
                for line in tqdm(in_f, desc=f"\tRunning thru file {file_path} ..."):
                    line = json.loads(line)
                    if 'DBLP' in line['externalids'] and line['externalids']['DBLP']:
                        curr_title = line['title'].strip(".").lower()
                        if curr_title in titles:
                            index = None
                            for i in titles[curr_title]:
                                pub = pubs[i]
                                doi = get_doi(pub)
                                if doi and "doi" in line and line["doi"]:
                                    if doi == line["doi"]:
                                        index = i
                                        break
                                elif "year" in line and line['year'] is not None and de_list(pub["year"], True) == de_list(line["year"], True):
                                    if line["venue"] == "ArXiv":
                                        if pub["journal"] and de_list(pub["journal"]) == "CoRR":
                                            index = i
                                            break
                                    elif pub["journal"] and de_list(pub["journal"]) == "CoRR":
                                        continue
                                    else:
                                        index = i
                                        break
                            if index and index not in removed_indices:
                                for key in ['referencecount', 'citationcount', 'influentialcitationcount', 's2fieldsofstudy', 'corpusid']:
                                    pub[key] = line[key]
                                pub['semantic_schol_authors'] = line['authors']
                                json.dump(pub, out_f)
                                out_f.write("\n")
                                removed_indices.add(index)
            counter += 1
        for i, pub in enumerate(pubs):
            if i not in removed_indices:
                not_matched += 1
                json.dump(pub, out_f)
                out_f.write("\n")
    print(f"Finished. Could not match {not_matched}/{len(pubs)} publications. ")

    print('\tParse finish! File ai_matched.jsonl created!')


def match_semantic_scholar_abstracts() -> None:
    source = Path(DATA_PATH, 'ai_matched.jsonl')
    semantic_scholar_path = Path(DATA_PATH, 'semantic_scholar')
    citation_path = Path(semantic_scholar_path, 'semantic_citations.csv')
    target = Path(DATA_PATH, 'ai_dataset.jsonl')

    citations = defaultdict(list)
    with open(citation_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="\tReading semantic_scholar citations ..."):
            line = line.split(",")
            citations[line[0].strip()].append(line[1].strip())

    pubs = []
    with open(source, "r", encoding="utf-8") as f:
        for line in f:
            pubs.append(json.loads(line))
    corpusids = {x['corpusid']: i for i, x in enumerate(pubs) if 'corpusid' in x}

    files = [str(file_path) for file_path in Path.iterdir(semantic_scholar_path) if "abstracts-part" in str(file_path)]
    counter = 1
    not_matched = 0
    removed_indices = set()
    with open(target, 'w', encoding="utf-8") as out_f:
        for file_path in files:
            print()
            print("\tReading file ... (", str(counter), "/", str(len(files)), ")")
            with gzip.open(Path(file_path), 'rb') as in_f:
                for line in tqdm(in_f, desc=f"\tRunning thru file {file_path} ..."):
                    line = json.loads(line)
                    if line['corpusid'] in corpusids: 
                        index = corpusids[line['corpusid']]
                        pub = pubs[index]
                        pub['abstract'] = line['abstract']
                        if pub['key'] in citations:
                            pub['citations'] = citations[pub['key']]
                        json.dump(pub, out_f)
                        out_f.write("\n")
                        removed_indices.add(index)
        for i, pub in enumerate(pubs):
            if i not in removed_indices:
                if pub['key'] in citations:
                    pub['citations'] = citations[pub['key']]
                not_matched += 1
                json.dump(pub, out_f)
                out_f.write("\n")
    print(f"Finished. Could not match {not_matched}/{len(pubs)} publications. ")

    print('\tParse finish! File ai_dataset.jsonl created!')


def match_semantic_scholar_persons() -> None:
    source = Path(DATA_PATH, 'ai_dataset.jsonl')
    semantic_scholar_path = Path(DATA_PATH, 'semantic_scholar')
    persons_path = Path(DATA_PATH, 'persons.jsonl')
    country_domains = Path(DATA_PATH, 'country_domains.csv')
    target = Path(DATA_PATH, "persons_matched.jsonl")

    pub_authors = []
    with open(source, 'r') as f:
        for line in tqdm(f, desc="\tExtracting semantic scholar authors ..."):
            line = json.loads(line)
            if 'semantic_schol_authors' in line:
                pub_authors.append(([line['author'], line['semantic_schol_authors']]))
    matched_author_ids = dict()
    for pub in tqdm(pub_authors, desc="\tTrying to match authors ..."):
        if len(pub[0]) == len(pub[1]):
            for dblp_auth in pub[0]:
                if dblp_auth not in matched_author_ids:
                    dblp_name = ''.join([s for s in dblp_auth if not s.isdigit()]).strip().lower()
                    for schol_auth in pub[1]:
                        if dblp_name == schol_auth['name'].lower() or unidecode(dblp_name) == unidecode(schol_auth['name'].lower()):
                            matched_author_ids[dblp_auth] = schol_auth['authorId']
                        elif f'{dblp_name.split(" ")[0][0]}. {unidecode(dblp_name).split(" ")[-1]}' == f'{schol_auth["name"].split(" ")[0][0]}. {unidecode(schol_auth["name"]).split(" ")[-1]}':
                            matched_author_ids[dblp_auth] = schol_auth['authorId']

    country_domains = dict()
    with open(Path(DATA_PATH, "countries_domain.csv"), 'r') as f:
        for line in f:
            curr = [x.strip() for x in line.split(";")]
            country_domains[curr[0]] = curr[0]
            country_domains[curr[1]] = curr[0]
    
    matching_persons = {}
    other_persons = []
    with open(persons_path, 'r') as f:
        for idx, line in enumerate(tqdm(f, desc="\tReading persons ...")):
            curr = json.loads(line)
            curr['id'] = idx
            curr_author = curr['author'] if isinstance(curr['author'], list) else [curr['author']]
            curr['author'] = curr_author
            if 'note' in person and person['note'] is not None:
                affi = []
                countries = set()
                if isinstance(person['note'], str):
                    affi.append(person['note'])
                elif isinstance(person['note'], list):
                    for x in person['note']:
                        if isinstance(x, str):
                            affi.append(x)
                person['dblp_affiliations'] = affi
                for aff in affi:
                    splitted = aff.split(",")[-1].strip()
                    if splitted in country_domains:
                        countries.add(country_domains[splitted])
                person['countries'] = list(countries)
                del person['note']
            for author in curr['author']:
                if author in matched_author_ids:
                    curr['sem_schol_id'] = matched_author_ids[author]
            if 'sem_schol_id' in curr:
                matching_persons[curr['sem_schol_id']] = curr
            else:
                other_persons.append(curr)

    files = [str(file_path) for file_path in Path.iterdir(semantic_scholar_path) if "authors-part" in str(file_path)]
    found_ids  = []
    counter = 1
    with open(target, 'w', encoding="utf-8") as out_f:
        for file_path in files:
            print()
            print("\tReading file ... (", str(counter), "/", str(len(files)), ")")
            with gzip.open(Path(file_path), 'rb') as in_f:
                for line in tqdm(in_f, desc=f"\tRunning thru file {file_path} ..."):
                    line = json.loads(line)
                    if line['authorid'] in matching_persons:
                        person = matching_persons[line['authorid']]
                        for key in ['affiliations', 'papercount', 'citationcount', 'hindex']:
                            person[f'schol_{key}'] = line[key]
                        json.dump(person, out_f)
                        out_f.write("\n")
                        found_ids.append(line['authorid'])
            counter += 1
        for idx, item in matching_persons.items():
            if idx not in found_ids:
                json.dump(item, out_f)
                out_f.write("\n")
        for person in other_persons:
            json.dump(person, out_f)
            out_f.write("\n")


def match_semantic_scholar_embeddings():
    semantic_scholar_path = Path(DATA_PATH, 'semantic_scholar')
    pubs_path = Path(DATA_PATH, 'ai_dataset.jsonl')
    target = Path(DATA_PATH, "sem_schol_embeddings.jsonl")

    pubs = {}
    with open(pubs_path, 'r') as f:
        for line in tqdm(f, desc="\tExtracting publications. "):
            line = json.loads(line)
            if 'corpusid' in line and int(line['year']) <= SEPARATION_YEAR:
                pubs[line['corpusid']] = line['key']

    files = [str(file_path) for file_path in Path.iterdir(semantic_scholar_path) if "embeddings-part" in str(file_path)]
    counter = 1
    with open(target, 'w', encoding="utf-8") as out_f:
        for file_path in files:
            print()
            print("\tReading file ... (", str(counter), "/", str(len(files)), ")")
            with gzip.open(Path(file_path), 'rb') as in_f:
                for line in tqdm(in_f, desc=f"\tRunning thru file {file_path} ..."):
                    line = json.loads(line)
                    if line['corpusid'] in pubs:
                        json.dump({'key': pubs[line['corpusid']], 'embedding': line['vector']}, out_f)
                        out_f.write("\n")
            counter += 1

    print(f'\tParse finish! File {target} created!')


def extract_persons(author_list: list) -> None:
    source = Path(DATA_PATH, 'dblp.xml')
    target = Path(DATA_PATH, 'persons.jsonl')
    
    print(f'\tParsing {source}')
    parse_dblp_person(source, target, author_list)
    print('   Parse finish! File persons.jsonl created!')
    print()
    

def helper_unlist(element):
    if isinstance(element, list):
        return element[0]
    return element


def parse_author_name(author_name) -> str:
    return ''.join([i for i in author_name if not i.isdigit()]).strip()


def create_graph_embeddings() -> None:
    from deepwalk.__main__ import process
    class myArgs:
        def __init__(self, format: str, input: str, output: str) -> None:
            self.format = format
            self.input = input
            self.output = output
            self.undirected = True
            self.seed = 0
            self.representation_size = 64
            self.number_walks = 10
            self.window_size = 5
            self.walk_length = 40
            self.max_memory_data_size = 1000000000
            self.workers = 1

    source_pubs = Path(DATA_PATH, 'ai_dataset.jsonl')
    source_persons = Path(DATA_PATH, 'persons_matched.jsonl')
    edgelist_file = Path(DATA_PATH, f"ai_dataset_{SEPARATION_YEAR}.edgelist")

    authors = dict()
    with open(source_persons, 'r') as f:
        for line in tqdm(f, desc="\tLoading authors..."):
            line = json.loads(line)
            curr_authors = line['author'] if isinstance(line['author'], list) else [line['author']]
            for author in curr_authors:
                authors[author] = line['id']

    with open(source_pubs, 'r') as f:
        with open(edgelist_file, "w") as out_f:
            for line in tqdm(f, desc="\tCreating edge list. "):
                line = json.loads(line)
                if int(line['year']) <= SEPARATION_YEAR:
                    auth = [authors[x] for x in line['author'] if x in authors]
                    curr_edges = [(auth[k], auth[v]) for k in range(len(auth)) for v in range(len(auth)) if k != v]
                    for edge in curr_edges:
                        out_f.write(f"{edge[0]} {edge[1]}\n")

    process(args=myArgs(format="edgelist", input=f"{str(edgelist_file)}", output=f"{str(DATA_PATH)}/graph.embeddings"))

    print("Done. Now execute 'deepwalk ...' ")


def create_scibert_paper_embeddings() -> None:
    from transformers import AutoTokenizer, AutoModel
    import torch
    source = Path(DATA_PATH, "ai_dataset.jsonl")
    target = Path(DATA_PATH, "semantic_embeddings.jsonl")
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Device is: {device}")
    tok = AutoTokenizer.from_pretrained("sentence-transformers/bert-base-nli-mean-tokens")
    model = AutoModel.from_pretrained("sentence-transformers/bert-base-nli-mean-tokens").to(device)

    # Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def calculate_embeddings(sentences: list) -> list:
        # Tokenize sentences
        encoded_input = tok(sentences, padding=True, truncation=True, max_length=128, return_tensors='pt')

        # Compute token embeddings
        with torch.no_grad():
            model_output = model(**encoded_input.to(device))

        # Perform pooling. In this case, mean pooling
        return mean_pooling(model_output, encoded_input['attention_mask']).cpu().numpy()
    
    def batch(iterable, n=1):
        l = len(iterable)
        for ndx in range(0, l, n):
            yield iterable[ndx:min(ndx + n, l)]

    # get sentences from pubs abstracts
    abstracts = []
    idx = 0
    with open(source, 'r') as f:
        for line in f:
            line = json.loads(line)
            if 'abstract' in line and int(line['year']) <= SEPARATION_YEAR:
                abstracts.append({'key': line['key'], 'abstract': line['abstract']})
                idx += 1

    batches = int(len(abstracts) / 128) + 1
    pbar = tqdm(total=batches, desc="\tCreating embeddings. ")
    with open(target, 'w', encoding='utf-8') as f:
        for batch in batch(abstracts, 128):
            curr_abstracts = [x['abstract'] for x in batch]
            embeddings = calculate_embeddings(curr_abstracts).astype(float)
            for curr_dict, embedding in zip(batch, embeddings):
                curr_dict['embedding'] = list(embedding)
                json.dump(curr_dict, f)
                f.write("\n")
            pbar.update(1)
    pbar.close()

    print("Finished extracting paper embeddings. ")


def create_scibert_author_embeddings(source_embeddings: Path, target: Path) -> None:
    source_paper = Path(DATA_PATH, "ai_dataset.jsonl")
    source_author = Path(DATA_PATH, "persons_matched.jsonl")

    authors = set()
    with open(source_author, 'r') as f:
        for line in f:
            line = json.loads(line)
            curr_author = line['author'] if isinstance(line['author'], list) else [line['author']]
            authors.update(curr_author)

    authorships = dict()
    with open(source_paper, 'r') as f:
        for line in tqdm(f, desc="\tLoading authorships ..."):
            line = json.loads(line)
            curr_authors = [x for x in line['author'] if x in authors]
            authorships[line['key']] = curr_authors

    author_embeddings = {x: [] for x in authors}
    with open(source_embeddings, 'r') as f:
        for line in tqdm(f, desc="\tLoading embeddings ..."):
            line = json.loads(line)
            for author in authorships[line['key']]:
                if author in author_embeddings:
                    if type(line['embedding']) == str:
                        author_embeddings[author].append(json.loads(line['embedding']))
                    else:
                        author_embeddings[author].append(line['embedding'])
                    
    with open(target, 'w') as f:
        for key, value in tqdm(author_embeddings.items(), desc="\tMeaning and saving author embeddings. "):
            if len(value) == 1:
                f.write(json.dumps({'author': key, 'embedding': value}) + "\n")
            elif len(value) != 0:
                f.write(json.dumps({'author': key, 'embedding': np.array(value).mean(axis=0).tolist()}) + "\n")

    print("Finished meaning semantic embeddings. ")


def extract_semantic_scholar_citations() -> None:
    """
    Extracts the Semantic Scholar citations from the downloaded data.
    :return: None
    """
    source = Path(DATA_PATH, "ai_dataset.jsonl")
    semantic_scholar_path = Path(DATA_PATH, 'semantic_scholar')
    target = Path(DATA_PATH, "semantic_citations.csv")

    pubs = {}
    with open(source, 'r') as f:
        for line in tqdm(f, desc="\tLoading publications. "):
            line = json.loads(line)
            if "corpusid" in line:
                pubs[line['corpusid']] = line['key']

    files = [str(file_path) for file_path in Path.iterdir(semantic_scholar_path) if "citations-part" in str(file_path)]
    counter = 1
    with open(target, 'w', encoding="utf-8") as out_f:
        for file_path in files:
            print()
            print("\tReading file ... (", str(counter), "/", str(len(files)), ")")
            with gzip.open(Path(file_path), 'rb') as in_f:
                for line in tqdm(in_f, desc=f"\tRunning thru file {file_path} ..."):
                    line = json.loads(line)
                    if line['citingcorpusid'] is not None and line['citedcorpusid'] is not None and int(line['citingcorpusid']) in pubs and int(line['citedcorpusid']) in pubs:
                        out_f.write(f"{pubs[int(line['citingcorpusid'])]},{pubs[int(line['citedcorpusid'])]}\n")
            counter += 1
    
    print("Finished extracting Semantic Scholar Citations. ")




def pipeline_prepare_db() -> None:
    """
    '*** Starting pipeline process to prepare airankings Database ***'
    :param db_type: Which type of DB
    :return: None
    """
    print('**** Starting pipeline process to prepare airankings Database ****')

    print('\nProcess 01 - Download DBLP data')
    # download_dblp()

    print('\nProcess 02 - Unzipping DBLP data')
    # unzip_dblp()

    print('\nProcess 03 - Create dblp.json')
    # extract_publications()

    print('\nProcess 04 - Create ai_article.json')
    # author_list = extract_ai_publications()

    print('\nProcess 05 - Create persons.jsonl')
    # extract_persons(author_list)

    print('\nProcess 06 - Match with Semantic Scholar Papers')
    # match_semantic_scholar_papers()

    print('\nProcess 07 - Extract citations')
    # extract_semantic_scholar_citations()

    print('\nProcess 08 - Match with Semantic Scholar Abstracts')
    match_semantic_scholar_abstracts()

    print('\nProcess 09 - Match persons to semantic scholar. ')
    # match_semantic_scholar_persons()

    print('\nProcess 10 - Extract semantic scholar embeddings. ')
    # match_semantic_scholar_embeddings()

    print('\nProcess 11 - Create graph embeddings')
    # create_graph_embeddings()

    print('\nProcess 12 - Create cognitive paper embeddings')
    # create_scibert_paper_embeddings()

    print('\nProcess 13 - Mean cognitive author embeddings')
    # create_scibert_author_embeddings(source_embeddings=Path(DATA_PATH, "semantic_embeddings.jsonl"), target=Path(DATA_PATH, "semantic_author_embeddings.jsonl"))
    # create_scibert_author_embeddings(source_embeddings=Path(DATA_PATH, "sem_schol_embeddings.jsonl"), target=Path(DATA_PATH, "sem_schol_author_embeddings.jsonl"))

    print('\n*** Pipeline process to prepare airankings Database Finished! ***')


if __name__ == '__main__':
    pipeline_prepare_db()
